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


inline double ExpConv( double dist, double delta )
{
  return exp( - dist / delta );
}


template <typename boxType>
void debug0( const int classID,
             const Bipartite &m_to_l,
             const int numL,
             const std::vector<int> &labeled,
             const boxType &box )
{

  std::unordered_set<int> hash;
  for ( int n = 0; n < numL; n++ ) {
    if ( classID == box.dataset.label[labeled[n]] ) {
      auto _to_l = m_to_l.getToSet( n );
      for ( auto& ele : _to_l ) {
        int l = ele.first;
        if ( hash.end() == hash.find( l ) ) {
          hash.insert( l );
        }
      }
    }
  }
  for ( auto& l : hash ) {
    DebugInfo( "Leaf ID: %d\n", l );
    DebugInfo( "estimated center" );
    algebra::emphasizeDim( &box.q[l*LabelSet::classes], LabelSet::classes, classID );

    auto _to_n = m_to_l.getFromSet( l );
    double vote[LabelSet::classes];
    algebra::zero( vote, LabelSet::classes );
    for ( auto& ele : _to_n ) {
      int n = ele.first;
      if ( n < numL ) {
        vote[box.dataset.label[labeled[n]]] += LabelSet::GetWeight( box.dataset.label[labeled[n]] );
      }
    }
    DebugInfo( "average labeled" );
    double s = algebra::sum_vec( vote, LabelSet::classes );
    scale( vote, LabelSet::classes, 1.0 / s );
    algebra::emphasizeDim( vote, LabelSet::classes, classID );

    char ch;
    scanf( "%c", &ch );
    if ( 'e' == ch ) break;
  }
  
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
  std::vector<int> estimated( M, -1 );
  for ( auto &ele : labeled ) {
    estimated[labeled[ele]] = box.dataset.label[labeled[ele]];
  }

  heap <double, std::pair<int,int>, false> priority;
  for ( int i=0; i<M-1; i++ ) {
    for ( int j=i+1; j<M; j++ ) {
      if ( -1 != estimated[i] && -1 != estimated[j] ) continue;
      priority.add( -algebra::dist_l2( box.dataset.feat[i],
                                       box.dataset.feat[j],
                                       box.dataset.dim ),
                    std::make_pair( i, j ) );
    }
    progress( i+1, M-1, "Constructing Heap" );
  }
  printf( "\n" );

  /* ---------- labeling ---------- */
  while ( ! priority.empty() ) {
    int pair = 
  }
  
  
  return 0;
}
