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


  // HOG parameters
  cvFeat<HOG>::options.orientation_bins = env["bins"].toInt();
  cvFeat<HOG>::options.cell_side = env["cell-size"].toInt();
  cvFeat<HOG>::options.enable_color = env["enable-color"].toInt();
  cvFeat<HOG>::options.gaussian_filter = env["gaussian-filter"].toInt();
  
  
  

  int i = 0;
  int n = static_cast<int>( imgList.size() );
  for ( auto& ele : imgList ) {
    album.push( std::move( cvFeat<HOG>::gen( ele ) ) );
    album.SetPatchSize( env["patch-sampling-size"].toInt() );
    album.SetPatchStride( env["patch-sampling-stride"].toInt() );
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
  
  RedBox<FeatImage<float>::PatchProxy,BinaryOnAxis> box;
  BuildDataset( album, lblList, box.feat, box.label, env["sampling-stride"], env["sampling-margin"] );

  int len = static_cast<int>( box.feat.size() );
  int dim = box.feat[0].dim();

  Info( "Feature Descriptor Dimension: %d", dim );

  
  WITH_OPEN( out, env["output"].c_str(), "wb" );
  fwrite( &len, sizeof(int), 1, out );
  fwrite( &dim, sizeof(int), 1, out );
  ProgressBar progressbar;
  progressbar.reset( len );
  int count = 0;
  for ( auto& p : box.feat ) {
    for ( int j=0; j<dim; j++ ) {
      float c = p[j];
      fwrite( &c, sizeof(float), 1, out );
    }
    progressbar.update( ++count, "dumping features" );
  }
  fwrite( &box.label[0], sizeof(int), len, out );
  END_WITH( out );

  Done( "%d descriptors written to %s.", len, env["output"].c_str() );
  return 0;
}
