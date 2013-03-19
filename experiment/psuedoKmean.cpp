#include <vector>
#include <string>
#include <unordered_set>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/random.hpp"
#include "LLPack/algorithms/algebra.hpp"
#include "../data/Label.hpp"
#include "../data/DataSet.hpp"
#include "../data/GrayBox.hpp"
#include "../optimize/power_solver.hpp"
#include "../optimize/TMeanShell.hpp"
#include "RanForest/RanForest.hpp"


using namespace EnvironmentVariable;
using namespace ran_forest;
using namespace cat_tree;

using ran_forest::Bipartite;

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

inline double entropy( Bipartite &m_to_l, const GrayBox<float,BinaryOnAxis> & box,
                       const std::vector<int> &labeled,
                       const std::vector<int> &unlabeled )
{
  int L = m_to_l.sizeB();
  int numL = static_cast<int>( labeled.size() );
  int count = 0;
  double d[LabelSet::classes];
  double E = 0.0;
  for ( int l=0; l<L; l++ ) {
    auto& _to_n = m_to_l.to( l );
    if ( 0 < _to_n.size() ) {
      double unit = 1.0 / _to_n.size();
      count++;
      algebra::zero( d, LabelSet::classes );
      for ( auto& ele : _to_n ) {
        int n = ele.first;
        if ( n < numL ) {
          d[box.dataset.label[labeled[n]]] += unit;
        } else {
          d[box.dataset.label[unlabeled[n]]] += unit;
        }
      }
      for ( int k=0; k<LabelSet::classes; k++ ) {
        if ( d[k] > 1e-5 ) {
          E -= d[k] * log( d[k] );
        }
      }
    }
  }
  return E / count;
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
      auto _to_l = m_to_l.from( n );
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

    auto _to_n = m_to_l.to( l );
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
  

  /* fill P */
  std::vector<double> P( numL * LabelSet::classes );
  memset( &P[0], 0, sizeof(double) * numL * LabelSet::classes );
  {
    int m = 0;
    for ( auto& ele : labeled ) {
      P[ m * LabelSet::classes + box.dataset.label[ele] ] = 1.0;
      m++;
    }
  }
  
  /* solve */

  int depth = box.forest.depth();
  Info( "Tree Depth: %d\n", depth );

  std::unique_ptr<double> y_normal(nullptr);  
  std::unique_ptr<double> y(nullptr);

  
  WITH_OPEN( out, "compare.txt", "w" );
  END_WITH( out );
  for ( int level=6; level<depth; level++ ) {
  
    Bipartite m_to_l = std::move( box.forest.batch_query( SubList::create( box.dataset.feat, training ),
                                                          level ) );

    DebugInfo( "entropy: %.5lf\n", entropy( m_to_l, box, labeled, unlabeled ) );

    WITH_OPEN( out, "compare.txt", "a" );
    fprintf( out, "level %d with %d leaves\n", level, box.forest.levelSize( level ) );

    PowerSolver solve;
    box.initVoters( inverseMap, m_to_l, numL );
    solve( numL, numU, &P[0], &m_to_l, &box.q[0], y_normal );

    {
      printf( "========== normal level %d ==========\n", level );
      int l_count = box.test( labeled, level );
      int u_count = box.test( unlabeled, level );
      Info( "all: %d/%ld (%.2lf)", l_count, labeled.size(), static_cast<double>( l_count * 100 ) / labeled.size() );
      Info( "all: %d/%ld (%.2lf)", u_count, unlabeled.size(), static_cast<double>( u_count * 100 ) / unlabeled.size() );
      // fprintf( out, "clustring off:  %.2lf\t%.2lf\n", 
      //          static_cast<double>( l_count * 100 ) / labeled.size(), 
      //          static_cast<double>( u_count * 100 ) / unlabeled.size() );
    }


    TMeanShell<float> shell;
    shell.Clustering( SubList::create( box.dataset.feat, training ), box.dataset.dim, m_to_l );
    DebugInfo( "entropy after clustering: %.5lf\n", entropy( m_to_l, box, labeled, unlabeled ) );

    box.initVoters( inverseMap, m_to_l, numL );
    double e = solve( numL, numU, &P[0], &m_to_l, &box.q[0], y );
    
    // debugging:
    fprintf( out, "(%.6lf)\n", e );

    // debugging:
    // debug0( 9, m_to_l, numL, labeled, box );
    // debug0( 4, m_to_l, numL, labeled, box );
    // debug0( 6, m_to_l, numL, labeled, box );
    


    {
      printf( "========== clustering level %d ==========\n", level );
      // int l_count = box.test( labeled, level );
      int l_count = box.test( labeled, m_to_l, 0, numL );
      int u_count = box.test( unlabeled, m_to_l, numL, M );
      
      Info( "all: %d/%ld (%.2lf)", l_count, labeled.size(), static_cast<double>( l_count * 100 ) / labeled.size() );
      Info( "all: %d/%ld (%.2lf)", u_count, unlabeled.size(), static_cast<double>( u_count * 100 ) / unlabeled.size() );
      // fprintf( out, "clustering on:   %.2lf\t%.2lf\n", 
      //          static_cast<double>( l_count * 100 ) / labeled.size(),
      //          static_cast<double>( u_count * 100 ) / unlabeled.size() );
      fprintf( out, "%.2lf\n%.2lf\n", static_cast<double>( l_count * 100 ) / labeled.size(),
               static_cast<double>( u_count * 100 ) / unlabeled.size() );
    }
    END_WITH( out );
  }

  return 0;
}
