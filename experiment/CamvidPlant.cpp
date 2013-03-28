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



void partition( std::vector<int> &label, std::vector<int> &fed,
                std::vector<int> &extended, float ratio = 0.5 )
{
  std::vector<std::vector<int> > classified;
  classified.resize( LabelSet::classes );
  for ( auto& ele : classified ) {
    ele.clear();
  }
  
  for ( auto& ele : fed ) {
    classified[label[ele]].push_back( ele );
  }

  fed.clear();
  extended.clear();


  
  for ( auto& item : classified ) {
    int n = static_cast<int>( item.size() );
    int k = static_cast<int>( n * ratio );
    std::vector<int> selected = rndgen::randperm( n, k );
    std::vector<int> mask( item.size(), 0 );
    for ( auto& ele : selected ) {
      mask[ele] = 1;
      fed.push_back( item[ele] );
    }
    
    for ( int i=0; i<n; i++ ) {
      if ( 0 == mask[i] ) {
        extended.push_back( item[i] );
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



  /* ---------- Load/Construct Forest ---------- */
  typename ran_forest::SimpleKernel<typename FeatImage<float>::PatchProxy, BinaryOnAxis>::Options options;

  options.dim = box.feat[0].dim();
  options.converge = 0.005;
  options.stopNum = 3;
  options.numHypo = 10;
    
  Forest<float,BinaryOnAxis> forest;
  forest.grow<ran_forest::SimpleKernel>( env["forest-size"],
                                         box.feat,
                                         options,
                                         env["propotion-per-tree"].toDouble() );

  // debugging:
  bool depth = forest.tree(0).reduce<bool>
    ( []( const ran_forest::Tree<float,BinaryOnAxis> &node,
          const std::vector<bool> &res )
      {
        return true;
      },
      []( const ran_forest::Tree<float,BinaryOnAxis> &leaf )
      {
        return true;
      } );

  DebugInfo( "%d vs %d", depth, forest.tree(0).depth() );


  
  Info( "Total Nodes:  %d", forest.nodeNum() );
  Info( "Total Leaves: %d", forest.levelSize( forest.depth() ) );
  Info( "Depth: %d", forest.depth() );

  forest.write( env["forest-dir"] );
  Done( "forest written to %s", env["forest-dir"].c_str() );

  return 0;
}
