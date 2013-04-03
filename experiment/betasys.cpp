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
#include "../data/BetaBox.hpp"
#include "../optimize/TMeanShell.hpp"


/* ----- used namespace ----- */
using namespace EnvironmentVariable;
using namespace ran_forest;
using namespace PatTk;
using namespace cat_tree;



const featEnum DESCRIPTOR = BGRF;
template <typename dataType>
using splitterType = BinaryOnDistance<dataType>;

void InitLabel()
{
  LabelSet::initialize( env["class-map"] );
  LabelSet::Summary();
}



void InitBox( std::vector<std::string>& imgList, std::vector<std::string>& lblList,
              Album<float>& album, BetaBox<FeatImage<float>::PatchProxy,splitterType> &box )
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
  
  lblList = std::move( path::FFFL( env["dataset"], imgList, "_L.png" ) );
  imgList = std::move( path::FFFL( env["dataset"], imgList, ".png" ) );

  ProgressBar progressbar;
  int N = static_cast<int>( imgList.size() );
  int margin = env["sampling-margin"].toInt();
  int stride = env["patch-stride"].toInt();
  box.trueLabel.clear();
  box.labeledp.clear();
  box.labeled.clear();
  box.unlabeled.clear();
  box.feat.clear();
  progressbar.reset( N );
  for ( int n=0; n<N; n++ ) {
    album.push( std::move( cvFeat<DESCRIPTOR>::gen( imgList[n] ) ) );
    progressbar.update( n+1, "Loading Album" );
  }
  album.SetPatchStride( 1 );
  album.SetPatchSize( env["patch-size"] );

  progressbar.reset( N );
  for ( int n=0; n<N; n++ ) {
    cv::Mat lbl = cv::imread( lblList[n] );
    for ( int i = margin; i < lbl.rows - margin; i += stride ) {
      for ( int j = margin; j < lbl.cols - margin; j += stride ) {
        // feature
        box.feat.push_back( album(n).Spawn( i, j ) );
        // true label
        box.trueLabel.push_back( LabelSet::GetClass( lbl.at<cv::Vec3b>( i, j )[0],
                                                     lbl.at<cv::Vec3b>( i, j )[1],
                                                     lbl.at<cv::Vec3b>( i, j )[2] ) );
        box.labeledp.push_back( isLabeled[n] );
        if ( isLabeled[n] ) {
          box.labeled.push_back( static_cast<int>( box.feat.size() ) - 1 );
        } else {
          box.unlabeled.push_back( static_cast<int>( box.feat.size() ) - 1 );
        }
      }
    }
    progressbar.update( n+1, "Preparing Data" );
  }

  printf( "\n" );
  // forest
  box.LoadForest( env["forest-dir"] );

}


void test( const Bipartite& graph, BetaBox<FeatImage<float>::PatchProxy,splitterType>& box,
           const std::vector<std::string>& imgList,
           std::string directory )
{
  /* Generate:
   * 1. Per Channel Under /per_channel
   * 2. labeled under /estimation
   * 3. reconstructed?
   */

  box.solve( graph );

  std::vector<int> correctCnt( imgList.size(), 0 );
  std::vector<int> totalCnt( imgList.size(), 0 );
  std::vector<std::vector<cv::Mat> > perCh( imgList.size() );
  std::vector<cv::Mat> est( imgList.size() );


  for ( int i=0; i<static_cast<int>( imgList.size() ); i++ ) {
    cv::Mat org = cv::imread( imgList[i] );
    perCh[i].resize( LabelSet::classes );
    for ( auto& ele : perCh[i] ) ele.create( org.rows, org.cols, CV_8UC1 );
    est[i].create( org.rows, org.cols, CV_8UC3 );
  }

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
    for ( int k=0; k<LabelSet::classes; k++ ) {
      perCh[id][k].at<uchar>( y, x ) = static_cast<uchar>( 255.0 * vote[k] );
    }

    int guess = std::distance( vote, std::max_element( vote, vote + LabelSet::classes ) );
    est[id].at<cv::Vec3b>( y, x )[0] = std::get<2>( LabelSet::GetColor( guess ) );
    est[id].at<cv::Vec3b>( y, x )[1] = std::get<1>( LabelSet::GetColor( guess ) );
    est[id].at<cv::Vec3b>( y, x )[2] = std::get<0>( LabelSet::GetColor( guess ) );

    totalCnt[id]++;
    if ( box.trueLabel[n] == guess ) {
      correctCnt[id]++;
    }

    progressbar.update( n+1, "Voting" );
  }
  printf( "\n" );

  // save out images
  std::vector<std::string> imgNames;
  imgNames.reserve( imgList.size() );
  for ( auto& ele : imgList ) imgNames.push_back( path::file( ele ) );
  
  system( strf( "rm -rf %s", directory.c_str() ).c_str() );
  system( strf( "mkdir -p %s/per_channel", directory.c_str() ).c_str() );
  system( strf( "mkdir -p %s/estimation", directory.c_str() ).c_str() );
  for ( int i=0; i<static_cast<int>( imgList.size() ); i++ ) {
    system( strf( "mkdir -p %s/per_channel/%s", directory.c_str(), imgNames[i].c_str() ).c_str() );
    for ( int k=0; k<LabelSet::classes; k++ ) {
      cv::imwrite( strf( "%s/per_channel/%s/%d.png",
                         directory.c_str(),
                         imgNames[i].c_str(),
                         k ),
                   perCh[i][k] );
    }
    cv::imwrite( strf( "%s/estimation/%s.png", directory.c_str(), imgNames[i].c_str() ), est[i] );
  }

  WITH_OPEN( out, strf( "%s/statics.txt", directory.c_str() ).c_str(), "w" );
  for ( int i=0; i<static_cast<int>( imgNames.size() ); i++ ) {
    fprintf( out, "correct: %d/%d (%.2lf%%)\n", correctCnt[i], totalCnt[i],
             static_cast<double>( correctCnt[i] ) * 100.0 / totalCnt[i] );
    Info( "%3d - correct: %d/%d (%.2lf%%)", i, correctCnt[i], totalCnt[i],
          static_cast<double>( correctCnt[i] ) * 100.0 / totalCnt[i] );
  }
  int allCnt = std::accumulate( correctCnt.begin(), correctCnt.end(), 0 );
  int allTot = std::accumulate( totalCnt.begin(), totalCnt.end(), 0 );
  fprintf( out, "average: %d/%d (%.2lf%%)\n", allCnt, allTot,
           static_cast<double>( allCnt ) * 100.0 / allTot );
  Info( "average: %d/%d (%.2lf%%)", allCnt, allTot,
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
  std::vector<std::string> imgList;
  std::vector<std::string> lblList;
  Album<float> album;
  BetaBox<FeatImage<float>::PatchProxy,splitterType> box;

  InitLabel();
  InitBox( imgList, lblList, album, box );
  

  /* ---------- Pure Random Forest ---------- */

  Info( "Leaves/Patches : %d/%d (%.2lf%%)",
        box.forest.levelSize( env["specified-level"] ),
        box.size(),
        static_cast<double>( box.forest.levelSize( env["specified-level"].toInt() ) )
        / box.size() * 100.0 );
        
  Info( "start testing." );
  if ( static_cast<std::string>( env["acquire-graph"] ) == "load" ) {
    Info( "Loading Prelim Graph ..." );
    Bipartite n_to_l( "prelim.graph" );
    test( n_to_l, box, imgList, env["output"] );
    Done( "Loading Graph" );
    TMeanShell<float> shell;
    shell.options.maxIter = 20;
    shell.options.replicate = env["replicate"];
    shell.Clustering( box.feat, box.dim(), n_to_l );
    test( n_to_l, box, imgList, env["output-knn"] );
  } else {
    Bipartite n_to_l = std::move( box.forest.batch_query( box.feat, env["specified-level"] ) );
    n_to_l.write( "prelim.graph" );
    test( n_to_l, box, imgList, env["output"] );
    TMeanShell<float> shell;
    shell.options.maxIter = 20;
    shell.options.replicate = env["replicate"];
    shell.Clustering( box.feat, box.dim(), n_to_l );
    test( n_to_l, box, imgList, env["output-knn"] );
  }
  return 0;
}
