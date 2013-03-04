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
  

  RedBox<FeatImage<float>::PatchProxy,BinaryOnAxis> box;
  BuildDataset( album, lblList, box.feat, box.label, env["sampling-margin"], env["sampling-stride"] );



  /* ---------- Load/Construct Forest ---------- */
  if ( 0 == strcmp( env["forest-source"].c_str(), "build" ) ||
       0 == strcmp( env["forest-source"].c_str(), "construct" ) ) {
    typename ran_forest::SimpleKernel<typename FeatImage<float>::PatchProxy, BinaryOnAxis>::Options options;

    options.dim = box.feat[0].dim();
    options.converge = 0.005;
    options.stopNum = 3;
    
    Forest<float,BinaryOnAxis> forest;
    forest.grow<ran_forest::SimpleKernel>( env["forest-size"],
                                           box.feat,
                                           options,
                                           env["propotion-per-tree"].toDouble() );

    Info( "Total Nodes:  %d", forest.nodeNum() );
    Info( "Total Leaves: %d", forest.levelSize( forest.depth() ) );

    forest.write( env["forest-dir"] );
    Done( "forest written to %s", env["forest-dir"].c_str() );

  }
    
  box.LoadForest( env["forest-dir"] );

  
  /* ---------- testing ---------- */
  WITH_OPEN( out, env["output"].c_str(), "w" );
  auto whole = rndgen::seq( static_cast<int>( box.feat.size() ) );
  for ( int level=0; level<box.forest.depth(); level++ ) {
    std::vector<int> count( box.forest.nodeNum(), 0 );
    for ( int i=0; i<static_cast<int>( box.feat.size() ); i++ ){
      auto res = box.forest.query( box.feat[i], level );
      for ( auto& item : res ) {
        box.q[LabelSet::classes * item + box.label[i] ] += 1.0;
        count[item]++;
      }
    }

    for ( int i=0; i<box.forest.nodeNum(); i++ ) {
      if ( 0 < count[i] ) {
        scale( &box.q[i*LabelSet::classes], LabelSet::classes, 1.0 / count[i] );
      }
    }

    int cnt = box.test( whole, level );

    Info( "level %d, (%.2lf\%) with %d leaves.", level, static_cast<double>( cnt * 100 ) / box.size(),
          box.forest.levelSize(level) );
    fprintf( out, "level %d, (%.2lf%%) with %d leaves.\n", level, static_cast<double>( cnt * 100 ) / box.size(), box.forest.levelSize(level) );
    fflush( out );
  }
  END_WITH( out );


  
  
}
