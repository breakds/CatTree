#include <vector>
#include <string>
#include <unordered_set>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/random.hpp"
#include "LLPack/algorithms/algebra.hpp"
#include "LLPack/algorithms/heap.hpp"
#include "../data/Label.hpp"
#include "../data/DataSet.hpp"
#include "../data/Bipartite.hpp"
#include "../data/GrayBox.hpp"
#include "../optimize/power_solver.hpp"
#include "../optimize/TMeanShell.hpp"
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


void InitHeap( int i, const GrayBox<float,BinaryOnAxis>& box, 
               const std::vector<int> &estimated,
               std::vector<heap<double, int, false> > &heaps )
{
  for ( int n=0; n<box.dataset.size(); n++ ) {
    if ( n == i ) continue;
    if ( -1 != estimated [n] ) continue;
    heaps[i].add( -algebra::dist_l2( &box.dataset.feat[i][0],
                                     &box.dataset.feat[n][0],
                                     box.dataset.dim ), n);
  }
}


int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (maybe  prim.conf?)" );
    exit( -1 );
  }

  env.parse( argv[1] );
  env.Summary();
  
  /* load label set */
  LabelSet::initialize( env["class-map"] );
  LabelSet::Summary();

  /* loading dataset and forest */
  GrayBox<float,BinaryOnAxis> box( env["feature-data-input"], env["forest-dir"] );
  printf( "nodeNum: %d\n", box.forest.nodeNum() );
  for ( int i=0; i<box.forest.depth(); i++ ) {
    printf( "%d: %d\n", i, box.forest.levelSize( i ) );
  }



  
  /* generate random partition */
  std::vector<int> testing = rndgen::seq( box.dataset.size() );
  std::vector<int> labeled;
  std::vector<int> unlabeled;
  partition( box.dataset, testing, labeled, env["testing-ratio"].toDouble() );
  partition( box.dataset, labeled, unlabeled, env["labeled-ratio"].toDouble() );
  std::vector<int> training = labeled;
  for ( auto& ele : unlabeled ) training.push_back( ele );

  Info( "labeled: %ld", labeled.size() );
  Info( "unlabeled: %ld", unlabeled.size() );
  Info( "testing: %ld", testing.size() );
  
  /* creating index inverse map */
  std::vector<int> inverseMap( box.dataset.size() );
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
  
  /* 1. numL = num of labeled features
   * 2. numU = num of unlabled features
   * 3. numV = num of voters
   * 4. m_to_l = patch -> voters
   * 5. pair_to_l = pair -> voters
   * 6. patchPairs = array of pairs
   * 7. w = pair weights
   * 8. P = labels for training patches
   * 9. q = initialization for voters
   */

  int numL = static_cast<int>( labeled.size() );
  int numU = static_cast<int>( unlabeled.size() );
  int M = numU + numL;

  /* ---------- Construct Heap ---------- */
  // estimated labels
  std::vector<int> estimated( M, -1 );
  // heap per datapoint
  std::vector<heap<double, int, false> > heaps( M );
  
  // final (overall) heap
  heap<double, int, false> finale;

  // labeled init
  {
    int count = 0;
      for ( auto &ele : labeled ) {
        estimated[ele] = box.dataset.label[ele];
        InitHeap( ele, box, estimated, heaps );
        finale.add( heaps[ele](0), ele );
        progress( ++count, numL, "init heaps" );
      }
      printf( "\n" );
  }
  
  /* ---------- Eject ---------- */
  {
    int remain = numU;
    while ( ! finale.empty() ) {
      int i = finale[0];
      finale.pop();
      int j = heaps[i][0];
      heaps[i].pop();
    

    
    
      if ( -1 == estimated[j] ) {
        remain --;
        progress( numU - remain, numU, "ejecting" );
        estimated[j] = estimated[i];
        if ( ! heaps[i].empty() ) {
          finale.add( heaps[i](0), i );
        }
        InitHeap( j, box, estimated, heaps );
        if ( ! heaps[j].empty() ) {
          finale.add( heaps[j](0), j );
        }
        if ( 0 == remain ) break;
      }
    }
    printf( "\n" );
  }

  /* ---------- Evaluation ---------- */
  DebugInfo( "here" );
  int count = 0;
  for ( auto& ele : unlabeled ) {
    if ( estimated[ele] == box.dataset.label[ele] ) {
      count++;
    }
  }
  
  Info( "accuracy unlabeld: (%d/%d) = %.2lf%%", count, numU, 
        static_cast<double>( count ) / numU * 100.0 );
  
  return 0;
}
