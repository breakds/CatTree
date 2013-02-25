#include <vector>
#include <string>
#include <tuple>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "../data/Label.hpp"
#include "../data/DataSet.hpp"


using namespace EnvironmentVariable;
using namespace cat_tree;

void BruteForceNN( DataSet<float>& dataset, std::vector<int>& idx,
                   std::vector<std::tuple<int,int,double> > &pairs )
{
  if ( idx.empty() ) {
    idx.resize( dataset.size() );
    for ( int i=0; i<dataset.size(); i++ ) {
      idx.push_back( i );
    }
  }
  for ( auto& i : idx ) {
    heap<double,int> ranker( env["k-for-knn"].toInt() );
    for ( auto& j : idx ) {
      if ( i == j ) continue;
      ranker.add( dist_l2( &dataset.feat[i][0], &dataset.feat[j][0], dataset.dim ), j );
    }
    for ( int j=0; j<ranker.len; j++ ) {
      pairs.push_back( std::make_tuple( i, ranker[j], ranker(j) ) );
    }
  }
}


int main( int argc, char **argv )
{
  if ( argc <2 ) {
    Error( "Missing configuration file in options. (maybe setup.conf?)" );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();
  

  LabelSet::initialize( env["class-map"] );
  LabelSet::Summary();

  std::vector<std::string> imgList = readlines( strf( "%s/list.txt", 
  						      env["dataset"].c_str()));
  
  /* ---------- load the features from .txt files ---------- */
  DataSet<float> dataset;
  int featDim = -1;
  int len = static_cast<int>( imgList.size() );
  int i = 0;
  for ( auto& img : imgList ) {
    WITH_OPEN( in, 
  	       strf( "%s/%s.txt", env["dataset"].c_str(), img.c_str() ).c_str(),
  	       "r" );
    std::vector<float> feat;
    if ( -1 != featDim ) {
      feat.reserve( featDim );
    }
    float tmp = 0.0f;
    int r = fscanf( in, "%f", &tmp );
    while ( 0 < r ) {
      feat.push_back( tmp );
      r = fscanf( in, "%f", &tmp );
    }
    featDim = static_cast<int>( feat.size() );
    dataset.push( feat, LabelSet::GetClass( img ) );
    END_WITH( in );
    i++;
    if ( 0 == i % 100 || len == i ) {
      progress( i, len, "Reading/Writing Features" );
    }
  }
  printf( "\n" );
  dataset.write( env["feature-data-output"].c_str() );
  Done( "written to %s.", env["feature-data-output"].c_str() );

  /*
  if ( 0 != env["generate-knn-pairs"].toInt() ) {
  std::vector<std::tuple<int,int,double> > pairs;
    pairs.clear();
    for ( int i=0; i<dataset.size()-1; i++ ) {
      heap<double,int> ranker( env["k-for-knn"].toInt() );
      for ( int j=i+1; j<dataset.size(); j++ ) {
        ranker.add( dist_l2( &dataset.feat[i][0], &dataset.feat[j][0], dataset.dim ), j );
      }
      for ( int j=0; j<ranker.len; j++ ) {
        pairs.push_back( std::make_tuple( i, ranker[j], ranker(j) ) );
      }
      
      if ( i == dataset.size() - 1 || 0 == i % 100 ) {
        progress( i + 1, dataset.size() - 1, "knn" );
      }
    }
    printf( "\n" );
    

    WITH_OPEN( out, env["knn-output"].c_str(), "w" );
    int pairNum = static_cast<int>( pairs.size() );
    fwrite( &pairNum, sizeof(int), 1, out );
    for ( auto& t : pairs ) {
      fwrite( &std::get<0>( t ), sizeof(int), 1, out );
      fwrite( &std::get<1>( t ), sizeof(int), 1, out );
      fwrite( &std::get<2>( t ), sizeof(double), 1, out );
    }
    END_WITH( out );
    Done( "write nearest pairs." );
  }
  */

  if ( 0 != env["feature-data-output"].c_str() ) {
    std::vector<std::tuple<int,int,double> > pairs;
    pairs.clear();
    std::vector<std::vector<int> > categories( LabelSet::classes );
    for ( int i=0; i<dataset.size(); i++ ) {
      categories[dataset.label[i]].push_back( i );
    }
    for ( int j=0; j<LabelSet::classes; j++ ) {
      BruteForceNN( dataset, categories[j], pairs );
    }
    WITH_OPEN( out, env["knn-output"].c_str(), "w" );
    int pairNum = static_cast<int>( pairs.size() );
    fwrite( &pairNum, sizeof(int), 1, out );
    for ( auto& t : pairs ) {
      fwrite( &std::get<0>( t ), sizeof(int), 1, out );
      fwrite( &std::get<1>( t ), sizeof(int), 1, out );
      fwrite( &std::get<2>( t ), sizeof(double), 1, out );
    }
    END_WITH( out );
    Done( "write nearest pairs." );
  }
  return 0;
}
