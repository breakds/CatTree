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



/* ----- used namespace ----- */
using namespace EnvironmentVariable;
using namespace ran_forest;
using namespace PatTk;
using namespace cat_tree;



const featEnum DESCRIPTOR = BGRF;
template <typename dataType>
using kernelType = VP<dataType>;

void InitLabel()
{
  LabelSet::initialize( env["class-map"] );
  LabelSet::Summary();
}



void InitBox( std::vector<std::string>& imgList, std::vector<std::string>& lblList,
              Album<float>& album, BetaBox<FeatImage<float>::PatchProxy,kernelType> &box )
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
  size_t N = imgList.size();
  int margin = env["sampling-margin"].toInt();
  int stride = env["patch-stride"].toInt();
  box.trueLabel.clear();
  box.labeledp.clear();
  box.labeled.clear();
  box.unlabeled.clear();
  box.feat.clear();
  progressbar.reset( N );
  cvFeat<HOG>::options.cell_side = env["cell-size"].toInt();
  for ( size_t n=0; n<N; n++ ) {
    album.push( std::move( cvFeat<DESCRIPTOR>::gen( imgList[n] ) ) );
    progressbar.update( n+1, "Loading Album" );
  }
  album.SetPatchStride( env["cell-stride"].toInt() );
  album.SetPatchSize( env["patch-size"].toInt() );

  progressbar.reset( N );
  for ( size_t n=0; n<N; n++ ) {
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
          box.labeled.push_back( box.feat.size() - 1 );
        } else {
          box.unlabeled.push_back( box.feat.size() - 1 );
        }
      }
    }
    progressbar.update( n+1, "Preparing Data" );
  }

  printf( "\n" );
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
  BetaBox<FeatImage<float>::PatchProxy,kernelType> box;

  InitLabel();
  InitBox( imgList, lblList, album, box );
  
  /* ---------- Tree Construction ---------- */
  VP<float>::Options options;
  options.converge = 9 / 255.0 * box.dim();
  options.proportion = 1.1;
  options.stopNum = 20;

  {
    box.forest.grow( env["forest-size"].toInt(), box.feat, box.dim(), options );
  }
  box.forest.write( env["forest-dir"] );
  
  return 0;
}
