#include <vector>
#include <string>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/pathname.hpp"
#include "../data/Label.hpp"
#include "PatTk/data/FeatImage.hpp"
#include "PatTk/interfaces/opencv_aux.hpp"
#include "RanForest/RanForest.hpp"
#include "../data/FingerBox.hpp"
#include "../optimize/TMeanShell.hpp"


/* ----- used namespace ----- */
using namespace EnvironmentVariable;
using namespace ran_forest;
using namespace PatTk;
using namespace cat_tree;



void InitLabel()
{
  LabelSet::initialize( env["class-map"] );
  LabelSet::Summary();
}



void InitBox( std::vector<std::string>& imgList, std::vector<std::string>& lblList,
              Album<float>& album, FingerBox &box )
{

  std::vector<std::string> trainList = std::move( readlines( strf( "%s/%s",
                                                                   env["dataset"].c_str(),
                                                                   env["training"].c_str() ) ) );
  std::vector<std::string> testList = std::move( readlines( strf( "%s/%s",
                                                                  env["dataset"].c_str(),
                                                                  env["testing"].c_str() ) ) );

  // insert into imgList
  std::vector<bool> isLabeled;
  isLabeled.reserve( trainList.size() + testList.size() );
  imgList.clear();
  imgList.reserve( trainList.size() + testList.size() );

  for ( auto& ele : trainList ) {
    imgList.push_back( ele );
    isLabeled.push_back( true );
  }

  for ( auto& ele : testList ) {
    imgList.push_back( ele );
    isLabeled.push_back( false );
  }
  
  lblList = std::move( path::FFFL( env["dataset"], imgList, ".txt" ) );
  imgList = std::move( path::FFFL( env["dataset"], imgList, ".png" ) );


  int N = static_cast<int>( imgList.size() );
  ProgressBar progressbar;
  progressbar.reset( N );
  box.feat.resize( N );
  box.trueLabel.resize( N );
  box.labeledp.resize( N );
  for ( int n=0; n<N; n++ ) {
    cv::Mat img = cv::imread( imgList[n] );
    
    std::vector<float> feat;
    box.feat[n].reserve( img.cols * img.rows * 3 );
    for ( int i=0; i<img.rows; i++ ) {
      for ( int j=0; j<img.cols; j++ ) {
        for ( int c=0; c<3; c++ ) {
          box.feat[n].push_back( static_cast<float>( img.at<cv::Vec3b>( i, j )[c] ) );
        }
      }
    }
    
    WITH_OPEN( in, lblList[n].c_str(), "r" );
    FSCANF_CHECK( "%d", &box.trueLabel[n] );
    END_WITH( in );

    if ( isLabeled[n] ) {
      box.labeledp[n] = true;
      box.labeled.push_back( n );
    } else {
      box.labeledp[n] = false;
      box.unlabeled.push_back( n );
    }
    progressbar.update( n+1, "Initializing Box" );
  }
  Done( "Box Loaded." );
  
  printf( "\n" );
  // forest
  box.LoadForest( env["forest-dir"] );
}

void GraphSummary( const Bipartite& graph, const FingerBox &box,
		   FILE* out )
{
  int L = graph.sizeB();
  int validCnt = 0;
  int labeledCnt = 0;
  for ( int l=0; l<L; l++ ) {
    auto& _to_n = graph.to( l );
    if ( 0 < _to_n.size() ) {
      validCnt++;
      bool flag = false;
      for ( auto& ele : _to_n ) {
	if ( box.labeledp[ele.first] ) flag = true;
      }
      if ( flag ) {
        labeledCnt++;
      }
    }
  }
  Info( "Labeled/Valid: %d/%d (%.2lf%%)", labeledCnt, validCnt,
	static_cast<double>( labeledCnt ) * 100.0 / validCnt );
  fprintf( out, "Labeled/Valid: %d/%d (%.2lf%%)", labeledCnt, validCnt,
	   static_cast<double>( labeledCnt ) * 100.0 / validCnt );
}


void test( const Bipartite& graph, FingerBox& box,
           const std::vector<std::string>& imgList,
           std::string directory )
{
  /* Generate:
   * 1. Per Channel Under /per_channel
   * 2. labeled under /estimation
   * 3. reconstructed?
   */

  box.solve( graph, 20 );

  std::vector<int> correctCnt( imgList.size(), 0 );
  std::vector<int> totalCnt( imgList.size(), 0 );



  // voting
  double vote[LabelSet::classes];
    
  ProgressBar progressbar;
  progressbar.reset( box.size() );
  for ( int n=0; n<box.size(); n++ ) {
    // get vote
    algebra::zero( vote, LabelSet::classes );
    auto& _to_l = graph.from( n );
    for ( auto& ele : _to_l ) {
      int l = ele.first;
      double alpha = ele.second;
      algebra::addScaledTo( vote, &box.q[l][0], LabelSet::classes, alpha );
    }

    int id = box.feat[n].id();
    int y = box.feat[n].y;
    int x = box.feat[n].x;
    
    int guess = std::distance( vote, std::max_element( vote, vote + LabelSet::classes ) );
    
    totalCnt[id]++;
    if ( box.trueLabel[n] == guess ) {
      correctCnt[id]++;
    }

    progressbar.update( n+1, "Voting" );
  }
  printf( "\n" );

  WITH_OPEN( out, strf( "%s/result.txt", env["output"].c_str() ).c_str(), "w" );
  for ( int n=0; n<box.size(); n++ ) {
    fprintf( out, "%d: %d/%d (%.2lf%%)\n", n, correctCnt[n], totalCnt[n],
             static_cast<double>( correctCnt[n] ) * 100.0 / totalCnt[n] );
  }
  int allCnt = std::accumulate( correctCnt.begin(), correctCnt.end(), 0 );
  int allTot = std::accumulate( totalCnt.begin(), totalCnt.end(), 0 );
  fprintf( out, "%d: %d/%d (%.2lf%%)\n", n, allCnt, allTot,
           static_cast<double>( allCnt ) * 100.0 / allTot );
  END_WITH( out );
  
  Done( "Write result to %s", directory.c_str() );
}


int main( int argc, char **argv )
{

  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (beta.conf?)" );
    exit( -1 );
  }
  
  env.parse( argv[1] );
  env.Summary();

  /* ---------- Initialization ---------- */
  // std::vector<std::string> imgList;
  // std::vector<std::string> lblList;
  // Album<float> album;
  // FingerBox box;

  InitLabel();
  // InitBox( imgList, lblList, album, box );
  
  /* ---------- Pure Random Forest ---------- */

  // Info( "Leaves/Patches : %d/%d (%.2lf%%)",
  //       box.forest.levelSize( env["specified-level"].toInt() ),
  //       box.size(),
  //       static_cast<double>( box.forest.levelSize( env["specified-level"].toInt() ) )
  //       / box.size() * 100.0 );
  // DebugInfo( "dim = %d", album(0).GetPatchDim() );
  // DebugInfo( "dim: %d", box.dim() );
  // Info( "start testing." );
  // if ( static_cast<std::string>( env["acquire-graph"] ) == "load" ) {
  //   Info( "Loading Prelim Graph ..." );
  //   Bipartite n_to_l( "prelim.graph" );
  //   Done( "Loading Graph" );
  //   test( n_to_l, box, imgList, env["output"] );
  //   TMeanShell<float> shell;
  //   shell.options.maxIter = 20;
  //   shell.options.replicate = env["replicate"].toInt();
  //   shell.options.wtBandwidth = env["dist-bandwidth"].toDouble();
  //   shell.Clustering( box.feat, box.dim(), n_to_l );
  //   test( n_to_l, box, imgList, env["output-knn"] );
  // } else {
  //   Bipartite n_to_l = std::move( box.forest.batch_query( box.feat, env["specified-level"] ) );
  //   n_to_l.write( "prelim.graph" );
  //   test( n_to_l, box, imgList, env["output"] );
  //   TMeanShell<float> shell;
  //   shell.options.maxIter = 20;
  //   shell.options.replicate = env["replicate"].toInt();
  //   shell.Clustering( box.feat, box.dim(), n_to_l );
  //   test( n_to_l, box, imgList, env["output-knn"] );
  // }
  // return 0;
}
