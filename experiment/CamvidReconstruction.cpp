#include <vector>
#include <string>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/utils/pathname.hpp"
#include "LLPack/algorithms/random.hpp"
#include "LLPack/algorithms/algebra.hpp"
#include "PatTk/data/Label.hpp"
#include "PatTk/data/FeatImage.hpp"
#include "PatTk/interfaces/opencv_aux.hpp"
#include "RanForest/RanForest.hpp"
#include "../data/RedBox.hpp"
#include "../optimize/power_solver.hpp"
#include "../optimize/TMeanShellFancy.hpp"

using namespace EnvironmentVariable;
using namespace ran_forest;
using namespace PatTk;
using namespace cat_tree;

class Pastable
{
private:

  std::vector<cv::Mat> imgs;
  std::vector<cv::Mat> cnts; 


public:

  Pastable() : imgs(), cnts() {}

  template <typename T>
  explicit Pastable( const Album<T>& album ) : imgs( album.size() ), cnts( album.size() )
  {
    for ( int i=0; i<album.size(); i++ ) {
      imgs[i] = cv::Mat::zeros( album(i).rows, album(i).cols, CV_32FC3 );
      cnts[i] = cv::Mat::zeros( album(i).rows, album(i).cols, CV_32FC1 );
    }
  }

  inline void clear()
  {
    for ( auto& ele : imgs ) std::fill( ele.begin<float>(), ele.end<float>(), 0.0f );
    for ( auto& ele : cnts ) std::fill( ele.begin<float>(), ele.end<float>(), 0.0f );
  }

  inline void paste( int id, int i, int j, int size, float alpha, float* content )
  {
    int radius = size >> 1;
    float *cp = content;
    for ( int di = -radius; di <= radius; di ++ ) {
      int y = i + di;
      for ( int dj = -radius; dj <= radius; dj ++ ) {
        int x = j + dj;
        if ( 0 <= y && y < cnts[id].rows &&
             0 <= x && x < cnts[id].cols ) {
          imgs[id].at<cv::Vec3f>( y, x )[0] += *(cp++) * alpha;
          imgs[id].at<cv::Vec3f>( y, x )[1] += *(cp++) * alpha;
          imgs[id].at<cv::Vec3f>( y, x )[2] += *(cp++) * alpha;
          cnts[id].at<float>( y, x ) += alpha;
        } else {
          cp += 3;
        }
      }
    }
  }


  inline void write( int id, std::string filename )
  {
    cv::Mat output = cv::Mat::zeros( cnts[id].rows, cnts[id].cols, CV_8UC3 );

    for ( int i=0; i<cnts[id].rows; i++ ) {
      for ( int j=0; j<cnts[id].cols; j++ ) {
        if ( 1e-5 < cnts[id].at<int>( i, j ) ) {
          for ( int c = 0; c < 3; c ++ ) {
            output.at<cv::Vec3b>( i, j )[c] = static_cast<uchar>( imgs[id].at<cv::Vec3f>( i, j )[c] / cnts[id].at<float>( i, j ) * 255.0 );
          }
        }
      }
    }
    cv::imwrite( filename, output );
  }
};

void init( std::vector<std::string>& imgList, std::vector<std::string>& lblList,
           Album<float>& album,
           Album<float>& bgrAlbum )
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
    bgrAlbum.push( std::move( cvFeat<BGR_FLOAT>::gen( ele ) ) );
    progress( ++i, n, "Loading Album" );
  }
  bgrAlbum.SetPatchSize( env["paste-size"].toInt() );
  bgrAlbum.SetPatchStride( 1 );
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


Bipartite extracRepresentatives( const RedBox<FeatImage<float>::PatchProxy,BinaryOnDistance> &box,
                                 double th )
{
  int N = static_cast<int>( box.feat.size() );
  Bipartite n_to_l( N, 100 );
  std::vector<int> remain = std::move( rndgen::seq( N ) );
  
  int L = 0;
  int dim = box.feat[0].dim();
  
  ProgressBar progressbar;
  progressbar.reset( N );
  while ( ! remain.empty() ) {
    std::vector<int> hold;
    hold.reserve( remain.size() );
    int i = remain[rndgen::randperm( static_cast<int>( remain.size() ), 1 )[0]];
    n_to_l.grow_b( L + 1 );
    for ( auto& j : remain ) {
      double dist2 = 0.0;
      for ( int c=0; c<dim; c++ ) {
        double tmp = box.feat[i][c] - box.feat[j][c];
        dist2 += tmp * tmp;
      }
      if ( dist2 < th ) {
        n_to_l.add( j, L, 1.0 );
        continue;
      } else {
        hold.push_back( j );
      }
    }
    L++;
    remain = hold;
    progressbar.update( N - static_cast<int>( remain.size() ), 
                        "random clustering" );
    if ( 0 == L % 100 ) {
      printf( "L: %d\n", L );
    }
  }
  Done( "random clustering. total class num: %d", n_to_l.sizeB() );
  return n_to_l;
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
  Album<float> bgrAlbum;
  init( imgList, lblList, album, bgrAlbum );



  RedBox<FeatImage<float>::PatchProxy,BinaryOnDistance> box;
  BuildDataset( album, lblList, box.feat, box.label, env["sampling-stride"], env["sampling-margin"] );


  printf( "dim: %d\n", box.dim() );

  Forest<float,BinaryOnDistance> forest( env["forest-dir"] );
  
  /* ---------- Reconstruction ---------- */
  int depth = forest.depth();

  std::vector<std::unique_ptr<float> > bgrVoters( forest.nodeNum() );
  int bgrDim = bgrAlbum(0).GetPatchDim();
  for ( auto& ele : bgrVoters ) {
    ele.reset( new float[bgrDim] );
  }

  

  float vote[bgrDim];

  system( strf( "mkdir -p %s", env["output-dir"].c_str() ).c_str() );

  ProgressBar progressbar;


  DebugInfo( "level %d: %d\n", env["start-level"].toInt(), forest.levelSize( env["start-level"].toInt() ) );
  for ( int level = env["start-level"].toInt(); level < env["start-level"]+1; level += env["level-stride"].toInt() ) {
    // naive
    Bipartite n_to_l = std::move( forest.batch_query( box.feat, level ) );


    // Bipartite n_to_c = std::move( extracRepresentatives( box, 0.1 ) );
    // n_to_c.write( "n_to_c.dat" );
    // Done( "wrote" );


    // BGR voters
    int L = n_to_l.sizeB();
    progressbar.reset( L );
    for ( int l=0; l<L; l++ ) {
      auto& _to_n = n_to_l.to( l );
      algebra::zero( bgrVoters[l].get(), bgrDim );
      if ( 0 < _to_n.size() ) {
        double s = 0.0;
        for ( auto& ele : _to_n ) {
          const int &n = ele.first;
          const double& alpha = ele.second;
          s += alpha;
          bgrAlbum(box.feat[n].id()).FetchPatch( box.feat[n].y, box.feat[n].x, vote );
          algebra::addScaledTo( bgrVoters[l].get(), vote, bgrDim, static_cast<float>( alpha ) );
        }
        algebra::scale( bgrVoters[l].get(), bgrDim, static_cast<float>( 1.0 / s ) );
      }
      progressbar.update( l + 1, "calculating voters" );
    }

    // voting
    int N = n_to_l.sizeA();

    {
      Pastable board( album );

      progressbar.reset( N );
      for ( int n=0; n<N; n++ ) {
        auto& _to_l = n_to_l.from( n );
        if ( 0 < _to_l.size() ) {
          for ( auto& ele : _to_l ) {
            const int &l = ele.first;
            const double &alpha = ele.first;
            board.paste( box.feat[n].id(), box.feat[n].y, box.feat[n].x,
                         env["paste-size"].toInt(), alpha, bgrVoters[l].get() );
          }
        }
      }

      // output
      for ( int i=0; i<static_cast<int>( imgList.size() ); i++ ) {
        system( strf( "mkdir -p %s/%s", env["output-dir"].c_str(), imgList[i].c_str() ).c_str() );
        board.write( i, strf( "%s/%s/%d.png", env["output-dir"].c_str(), imgList[i].c_str(), level ) );
      }
    }


    // clustering
    TMeanShell<float> shell;
    shell.options.maxIter = 20;
    shell.options.replicate = env["replicate"];
    shell.Clustering( box.feat, box.dim(), n_to_l );

    // BGR voters
    L = n_to_l.sizeB();
    progressbar.reset( L );
    // debugging
    int validNodeCount = 0;
    for ( int l=0; l<L; l++ ) {
      auto& _to_n = n_to_l.to( l );
      algebra::zero( bgrVoters[l].get(), bgrDim );
      if ( 0 < _to_n.size() ) {
        double s = 0.0;
        for ( auto& ele : _to_n ) {
          const int &n = ele.first;
          const double& alpha = ele.second;
          s += alpha;
          bgrAlbum(box.feat[n].id()).FetchPatch( box.feat[n].y, box.feat[n].x, vote );
          algebra::addScaledTo( bgrVoters[l].get(), vote, bgrDim, static_cast<float>( alpha ) );
        }
        algebra::scale( bgrVoters[l].get(), bgrDim, static_cast<float>( 1.0 / s ) );
        // debugging:
        validNodeCount ++;
      }
      progressbar.update( l + 1, "calculating voters" );
    }

    // debugging:
    DebugInfo( "valid node #: %d\n", validNodeCount );

    // voting
    {
      Pastable board( album );
      N = n_to_l.sizeA();
      progressbar.reset( N );
      for ( int n=0; n<N; n++ ) {
        auto& _to_l = n_to_l.from( n );
        if ( 0 < _to_l.size() ) {
          for ( auto& ele : _to_l ) {
            const int &l = ele.first;
            const double &alpha = ele.first;
            board.paste( box.feat[n].id(), box.feat[n].y, box.feat[n].x,
                         env["paste-size"].toInt(), alpha, bgrVoters[l].get() );
          }
        }
      }

      // output
      for ( int i=0; i<static_cast<int>( imgList.size() ); i++ ) {
        system( strf( "mkdir -p %s/%s/compare", env["output-dir"].c_str(), imgList[i].c_str() ).c_str() );
        board.write( i, strf( "%s/%s/compare/%d.png", env["output-dir"].c_str(), imgList[i].c_str(), level ) );
      }
    }


  }

  return 0;
}
                                  
      

  
