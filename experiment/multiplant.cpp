#include <vector>
#include <string>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/pathname.hpp"
#include "PatTk/data/Label.hpp"
#include "PatTk/data/FeatImage.hpp"
#include "PatTk/interfaces/opencv_aux.hpp"
#include "RanForest/RanForest.hpp"


// namespaces:
using namespace EnvironmentVariable;
using namespace ran_forest;
using namespace PatTk;


// constant defnition
const featEnum DESCRIPTOR = BGRF;
template <typename dataType>
using kernelType = VP<dataType>;

int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options." );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();

  // The following code put all the image names into imgList

  std::vector<std::string> trainList = std::move( readlines( strf( "%s/%s",
                                                                   env["dataset"].c_str(),
                                                                   env["training"].c_str() ) ) );
  std::vector<std::string> testList = std::move( readlines( strf( "%s/%s",
                                                                  env["dataset"].c_str(),
                                                                  env["testing"].c_str() ) ) );
  
  std::vector<std::string> imgList;
  imgList.reserve( trainList.size() + testList.size() );

  for ( auto& ele : trainList ) {
    imgList.push_back( ele );
  }

  for ( auto& ele : testList ) {
    imgList.push_back( ele );
  }
  
  imgList = std::move( path::FFFL( env["dataset"], imgList, ".png" ) );


  // The following code read the album
  Album<float> album;
  ProgressBar progressbar;
  size_t N = imgList.size();
  int stride = env["patch-stride"].toInt();
  progressbar.reset( N );
  cvFeat<HOG>::options.cell_side = env["cell-size"].toInt();
  for ( size_t n=0; n<N; n++ ) {
    album.push( std::move( cvFeat<DESCRIPTOR>::gen( imgList[n] ) ) );
    progressbar.update( n+1, "Loading Album" );
  }
  album.SetPatchStride( env["cell-stride"].toInt() );
  album.SetPatchSize( env["patch-size"].toInt() );


  int margin = env["sampling-margin"].toInt();
  std::vector<std::vector<float> > vantages;

  // Stage 1
  {
    // Tree Construction Options
    VP<float>::Options options;
    // considered as converged when largets distance is less than this
    options.converge = 9 / 225.0 * album(0).GetPatchDim();
    // use 100% data to build the tree
    options.proportion = 1.1;
    // stop if node contains less than or equal to 5 data points
    options.stopNum = 5;

    
    std::vector<TMeanShell<float> > shell;
    std::vector<Bipartite> graph( album.size() );
    for ( auto __attribute__((__unused__)) &img : album ) {
      shell.emplace_back( album(0).GetPatchDim() );
    }

    progressbar.reset( album.size() );
    int finished = 0;

#   pragma omp parallel for
    for ( int i=0; i<album.size(); i++ ) {
      auto& img = album(i);
      // get all patches from img to feat
      std::vector<FeatImage<float>::PatchProxy> feat;
      for ( int i = margin; i < img.rows - margin; i += stride ) {
        for ( int j = margin; j < img.cols - margin; j += stride ) {
          feat.push_back( img.Spawn( i, j ) );
        }
      }

      // build forest
      Forest<float,VP> forest;
      // 1 tree, feature vectors in feat, feat[0].size as dimension, and options
      forest.grow( 5, feat, feat[0].size(), options );
      // query tree
      graph[i] = std::move( forest.batchQuery( feat ) );
      // KNN with TMeanShell -> n_to_l
      shell[i].options.maxIter = 20;
      shell[i].options.replicate = 2;
      shell[i].Clustering( feat, graph[i] );
    }

    for ( int i=0; i<album.size(); i++ ) {
      size_t L = graph[i].sizeB();
      for ( size_t l=0; l<L; l++ ) {
        auto& _to_n = graph[i].to( l );
        if ( !_to_n.empty() ) {
          vantages.push_back( shell[i].centers[l] );
        }
      }
    }
#   pragma omp critical
    {
      progressbar.update( ++finished, "stage 1 training" );
    }
  }
  

  // Stage 2
  // Tree Construction Options
  VP<float>::Options options;
  // considered as converged when largets distance is less than this
  options.converge = 9 / 225.0 * album(0).GetPatchDim();
  // use 100% data to build the tree
  options.proportion = 1.1;
  // stop if node contains less than or equal to 5 data points
  options.stopNum = 5;
  Forest<float,VP> forest;
  forest.grow( env["forest-size"], vantages, vantages[0].size(), options );
  Done( "Forest built." );
  forest.write( env["forest-dir"] );

  Done( "Forest written to %s.", env["forest-dir"].c_str() );
  return 0;
}





