#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/pathname.hpp"
#include "PatTk/data/Label.hpp"
#include "PatTk/data/FeatImage.hpp"
#include "PatTk/interfaces/opencv_aux.hpp"
#include "RanForest/RanForest.hpp"
#include "../data/RedBox.hpp"
#include "../optimize/power_solver.hpp"
#include "../optimize/TMeanShell.hpp"

using namespace EnvironmentVariable;
using namespace ran_forest;
using namespace PatTk;
using namespace cat_tree;

void init( std::vector<std::string>& imgList, std::vector<std::string>& lblList,
           Album<float>& album )
{
  LabelSet::initialize( env["class-map"] );
  LabelSet::Summary();
  imgList = std::move( readlines( strf( "%s/%s", env["dataset"].c_str(),
                                        env["list-file"].c_str() ) ) );
  lblList = std::move( path::FFFL( env["dataset"], imgList, "_L.png" ) );
  imgList = std::move( path::FFFL( env["dataset"], imgList, ".png" ) );

  int i = 0;
  int n = static_cast<int>( imgList.size() );
  for ( auto& ele : imgList ) {
    album.push( std::move( cvFeat<HOG>::gen( ele ) ) );
    progress( ++i, n, "Loading Album" );
  }
  printf( "\n" );
}

void BuildDataset( Album<float>& album,
                   const std::vector<std::string>& lblList,
                   std::vector<FeatImage<float>::PatchProxy>& feat,
                   std::vector<int>& label, int stride=1, int margin=7 )
{
  feat.clear();
  label.clear();

  for ( auto& img : album ) {
    for ( int i = margin; i < img.rows - margin; i += stride ) {
      for ( int j = margin; j < img.cols - margin; j += stride ) {
        feat.push_back( img.Spawn( i, j ) );
        
      }
    }
  }

  for ( auto& s : lblList ) {
    cv::Mat lbl = cv::imread( s );
    for ( int i = margin; i < lbl.rows - margin; i += stride ) {
      for ( int j = margin; j < lbl.cols - margin; j += stride ) {
        label.push_back( LabelSet::GetClass( lbl.at<cv::Vec3b>( i, j )[0],
                                             lbl.at<cv::Vec3b>( i, j )[1],
                                             lbl.at<cv::Vec3b>( i, j )[2] ) );
      }
    }
  }
}



int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (camvid.conf?)" );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();

  // initialize label set, image lists, album
  std::vector<std::string> imgList;
  std::vector<std::string> lblList;
  Album<float> album;
  init( imgList, lblList, album );



  RedBox<FeatImage<float>::PatchProxy,BinaryOnSubspace> box;
  BuildDataset( album, lblList, box.feat, box.label, env["sampling-margin"], env["sampling-stride"] );
  printf( "dim: %d\n", box.dim() );

  Forest<float,BinaryOnAxis> forest( env["forest-dir"] );


  /* ---------- Reconstruction ---------- */
  int depth = forest.depth();
  
  for ( int level = 4; level < depth; level++ ) {
    Bipartite n_to_l = std::move( forest.batch_query( box.feat, level ) );
    TMeanShell<float> shell;
    shell.Clustering( box.feat, box.dim(), n_to_l  );
  }

  return 0;
}
                                  
      

  
