#include <vector>
#include <string>
#include <unordered_set>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/random.hpp"
#include "../data/Label.hpp"
#include "../data/DataSet.hpp"
#include "../data/Bipartite.hpp"
#include "../data/GrayBox.hpp"
#include "../optimize/solver.hpp"
#include "RanForest/RanForest.hpp"

using namespace EnvironmentVariable;
using namespace ran_forest;
using namespace cat_tree; 

int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (maybe  exp.conf?)" );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();

  /* load label set */
  LabelSet::initialize( env["class-map"] );
  LabelSet::Summary();

  /* loading dataset and forest */
  GrayBox<float,BinaryOnAxis> box( env["feature-data-input"], env["forest-dir"] );

  std::vector<int> car; // id = 20
  std::vector<int> piano; // id = 47

  for ( int i=0; i<box.dataset.size(); i++ ) {
    if ( 20 == box.dataset.label[i] ) car.push_back( i );
    if ( 47 == box.dataset.label[i] ) piano.push_back( i );
  }

  std::vector<int> all;
  for ( auto& ele : car ) all.push_back( ele );
  for ( auto& ele : piano ) all.push_back( ele );

  
  Info( "car: %ld\n", car.size() );
  Info( "piano: %ld\n", piano.size() );

  WITH_OPEN( out, "visualize.txt", "w" );
  for ( auto& i : all ) {
    for ( auto& j : all ) {
      fprintf( out, "%.5lf\n", dist_l2( &box.dataset.feat[i][0], &box.dataset.feat[j][0], box.dataset.dim ) );
    }
  }
  END_WITH( out );
  return 0;
}
