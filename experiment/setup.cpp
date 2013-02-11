#include <vector>
#include <string>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "../data/Label.hpp"
#include "../data/DataSet.hpp"


using namespace EnvironmentVariable;
using namespace cat_tree;

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
  printf( "label=%d\n", dataset.label[0] );
  dataset.write( env["feature-data-output"].c_str() );
  Done( "written to %s.", env["feature-data-output"].c_str() );
  return 0;
}
