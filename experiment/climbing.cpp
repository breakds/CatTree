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

template <typename T>
void partition( DataSet<T> &data, std::vector<int> &fed,
                std::vector<int> &extended, float ratio = 0.5 )
{
  std::vector<std::vector<int> > classified;
  classified.resize( LabelSet::classes );
  for ( auto& ele : classified ) {
    ele.clear();
  }
  
  for ( auto& ele : fed ) {
    classified[data.label[ele]].push_back( ele );
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


inline double ExpConv( double dist, double delta )
{
  return exp( - dist / delta );
}



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
  GrayBox<float,BinaryOnSubspace> box( env["feature-data-input"], env["forest-dir"] );
  printf( "nodeNum: %d\n", box.forest.nodeNum() );
  
  box.zero();

  auto whole = rndgen::seq( box.dataset.size() );

  WITH_OPEN( out, "output.txt", "w" );
  for ( int level=0; level<box.forest.depth(); level++ ) {

    std::vector<int> count( box.forest.nodeNum(), 0 );
    
    for ( int i=0; i<box.dataset.size(); i++ ) {
      auto res = box.forest.query( box.dataset.feat[i], level );
      for ( auto& item : res ) {
        box.q[LabelSet::classes*item+box.dataset.label[i]] += 1.0;
        count[item]++;
      }
    }

    for ( int i=0; i<box.forest.nodeNum(); i++ ) {
      if ( 0 < count[i] ) {
        scale( &box.q[i*LabelSet::classes], LabelSet::classes, 1.0 / count[i] );
      }
    }
    
    int cnt = box.test( whole, level );
    Info( "level %d, (%.2lf\%) with %d leaves.", level, static_cast<double>( cnt * 100 ) / box.dataset.size(),
          box.forest.levelSize(level) );
    fprintf( out, "level %d, (%.2lf%%) with %d leaves.\n", level, static_cast<double>( cnt * 100 ) / box.dataset.size(), box.forest.levelSize(level) );
  }
  END_WITH( out );
  return 0;
}
