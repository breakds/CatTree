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
#include "../data/Bipartite.hpp"
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
  

  RedBox<FeatImage<float>::PatchProxy,BinaryOnSubspace> box;
  BuildDataset( album, lblList, box.feat, box.label, env["sampling-margin"], env["sampling-stride"] );



  /* ---------- Load/Construct Forest ---------- */
  if ( 0 == strcmp( env["forest-source"].c_str(), "build" ) ||
       0 == strcmp( env["forest-source"].c_str(), "construct" ) ) {
    typename ran_forest::MaxGapSubspaceKernel<typename FeatImage<float>::PatchProxy, BinaryOnSubspace>::Options options;

    options.dim = box.feat[0].dim();
    options.converge = 0.005;
    options.stopNum = 3;
    options.dimPrelim = 10;
    options.dimPrelim = 10;
    options.numHypo = 10;
    
    Forest<float,BinaryOnSubspace> forest;
    forest.grow<ran_forest::MaxGapSubspaceKernel>( env["forest-size"],
                                                   box.feat,
                                                   options,
                                                   env["propotion-per-tree"].toDouble() );

    Info( "Total Nodes:  %d", forest.nodeNum() );
    Info( "Total Leaves: %d", forest.levelSize( forest.depth() ) );

    forest.write( env["forest-dir"] );
    Done( "forest written to %s", env["forest-dir"].c_str() );

  }
    
  box.LoadForest( env["forest-dir"] );


  std::vector<int> testing = rndgen::seq( box.feat.size() );
  std::vector<int> labeled;
  std::vector<int> unlabeled;
  partition( box.label, testing, labeled, env["testing-ratio"].toDouble() );
  partition( box.label, labeled, unlabeled, env["labeled-ratio"].toDouble() );
  std::vector<int> training = labeled;
  for ( auto& ele : unlabeled ) training.push_back( ele );
  int numL = static_cast<int>( labeled.size() );
  int numU = static_cast<int>( unlabeled.size() );
  int M = numU + numL;
  Info( "labeled: %ld", labeled.size() );
  Info( "unlabeled: %ld", unlabeled.size() );
  Info( "testing: %ld", testing.size() );
  
  /* creating index inverse map */
  std::vector<int> inverseMap( box.feat.size() );
  {
    int i = 0;
    for ( auto& ele : labeled ) {
      inverseMap[ele] = i++;
    }
    for ( auto& ele : unlabeled ) {
      inverseMap[ele] = i++;
    }
    for ( auto& ele : testing ) {
      inverseMap[ele] = -1;
    }
  }



  /* ---------- testing ---------- */
  // WITH_OPEN( out, env["output"].c_str(), "w" );
  // auto whole = rndgen::seq( static_cast<int>( box.feat.size() ) );
  // for ( int level=0; level<box.forest.depth(); level++ ) {
  //   std::vector<int> count( box.forest.nodeNum(), 0 );
  //   for ( int i=0; i<static_cast<int>( box.feat.size() ); i++ ){
  //     auto res = box.forest.query( box.feat[i], level );
  //     for ( auto& item : res ) {
  //       box.q[LabelSet::classes * item + box.label[i] ] += 1.0;
  //       count[item]++;
  //     }
  //   }

  //   for ( int i=0; i<box.forest.nodeNum(); i++ ) {
  //     if ( 0 < count[i] ) {
  //       scale( &box.q[i*LabelSet::classes], LabelSet::classes, 1.0 / count[i] );
  //     }
  //   }

  //   int cnt = box.test( whole, level );

  //   Info( "level %d, (%.2lf\%) with %d leaves.", level, static_cast<double>( cnt * 100 ) / box.size(),
  //         box.forest.levelSize(level) );
  //   fprintf( out, "level %d, (%.2lf%%) with %d leaves.\n", level, static_cast<double>( cnt * 100 ) / box.size(), box.forest.levelSize(level) );
  //   fflush( out );
  // }
  // END_WITH( out );



  /* fill P */
  std::vector<double> P( numL * LabelSet::classes );
  memset( &P[0], 0, sizeof(double) * numL * LabelSet::classes );
  {
    int m = 0;
    for ( auto& ele : labeled ) {
      P[ m * LabelSet::classes + box.label[ele] ] = 1.0;
      m++;
    }
  }

  

  // reset file
  WITH_OPEN( out, "camvid.out.txt", "w" );
  END_WITH( out );
  /* ---------- testing ---------- */
  int depth = box.forest.depth();
  for ( int level=5; level<depth; level++ ) {
    Bipartite m_to_l( M, box.forest.nodeNum() );
    {
      int m = 0;
      for ( auto& ele : labeled ) {
        auto res = box.forest.query( box.feat[ele], level );
        double alpha = 1.0 / res.size();
        for ( auto& item : res ) {
          m_to_l.add( m, item, alpha );
        }
        m++;
        if ( 0 == m % 100 ) {
          progress( m, labeled.size() + unlabeled.size(), "query" );
        }
      }

      for ( auto& ele : unlabeled ) {
        auto res = box.forest.query( box.feat[ele], level );
        double alpha = 1.0 / res.size();
        for ( auto& item : res ) {
          m_to_l.add( m, item, alpha );
        }
        m++;
        if ( 0 == m % 100 ) {
          progress( m, labeled.size() + unlabeled.size(), "query" );
        }
      }
      progress( 1, 1, "query" );
      printf( "\n" );
    }

    
    WITH_OPEN( out, "camvid.out.txt", "a" );
    fprintf( out, "level %d with %d leaves\n", level, box.forest.levelSize( level ) );
    PowerSolver solve;
    box.initVoters( inverseMap, m_to_l, numL );
    solve( numL, numU, &P[0], &m_to_l, &box.q[0] );

    {
      printf( "========== normal level %d ==========\n", level );
      int l_count = box.test( labeled, level );
      int u_count = box.test( unlabeled, level );
      Info( "all: %d/%ld (%.2lf)", l_count, labeled.size(), static_cast<double>( l_count * 100 ) / labeled.size() );
      Info( "all: %d/%ld (%.2lf)", u_count, unlabeled.size(), static_cast<double>( u_count * 100 ) / unlabeled.size() );
      fprintf( out, "clustring off:  %.2lf\t%.2lf\n", 
               static_cast<double>( l_count * 100 ) / labeled.size(), 
               static_cast<double>( u_count * 100 ) / unlabeled.size() );
    }
    

    TMeanShell<float> shell;
    shell.Clustering( box.feat, training, box.dim(), m_to_l );

    box.initVoters( inverseMap, m_to_l, M );
    solve( numL, numU, &P[0], &m_to_l, &box.q[0] );

    {
      printf( "========== clustering level %d ==========\n", level );
      // int l_count = box.test( labeled, level );
      int l_count = box.test( labeled, m_to_l, 0, numL );
      int u_count = box.test( unlabeled, m_to_l, numL, M );
      
      Info( "all: %d/%ld (%.2lf)", l_count, labeled.size(), static_cast<double>( l_count * 100 ) / labeled.size() );
      Info( "all: %d/%ld (%.2lf)", u_count, unlabeled.size(), static_cast<double>( u_count * 100 ) / unlabeled.size() );
      fprintf( out, "clustering on:   %.2lf\t%.2lf\n", 
               static_cast<double>( l_count * 100 ) / labeled.size(),
               static_cast<double>( u_count * 100 ) / unlabeled.size() );
      END_WITH( out );
    }
    
  }
  

    
  

  return 0;
  
}
