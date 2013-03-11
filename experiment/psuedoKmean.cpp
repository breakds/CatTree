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
#include "../optimize/power_solver.hpp"
#include "../optimize/TMeans.hpp"
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
  
  
  WITH_OPEN( out, "compare.txt", "w" );
  for ( int level=4; level<14; level++ ) {
  
    /* forest query */
    Bipartite m_to_l( M, box.forest.nodeNum() );
    {
      int m = 0;
      for ( auto& ele : labeled ) {
        auto res = box.forest.query( box.dataset.feat[ele], level );
        double alpha = 1.0 / res.size();
        for ( auto& item : res ) {
          m_to_l.add( m, item, alpha );
        }
        m++;
        if ( 0 == m % 100 ) {
          progress( m, labeled.size() + unlabeled.size(), "query" );
        }
      }

      for ( auto& ele : unlabeled ) {
        auto res = box.forest.query( box.dataset.feat[ele], level );
        double alpha = 1.0 / res.size();
        for ( auto& item : res ) {
          m_to_l.add( m, item, alpha );
        }
        m++;
        if ( 0 == m % 100 ) {
          progress( m, labeled.size() + unlabeled.size(), "query" );
        }
      }
      progress( 1, 1, "query" );
      printf( "\n" );
    }

    PowerSolver solve;
    box.initVoters( inverseMap, m_to_l, numL );
    solve( numL, numU, &P[0], &m_to_l, &box.q[0] );

    {
      printf( "========== normal level %d ==========\n", level );
      int l_count = box.test( labeled, level );
      int u_count = box.test( unlabeled, level );
      Info( "all: %d/%ld (%.2lf)", l_count, labeled.size(), static_cast<double>( l_count * 100 ) / labeled.size() );
      Info( "all: %d/%ld (%.2lf)", u_count, unlabeled.size(), static_cast<double>( u_count * 100 ) / unlabeled.size() );
      fprintf( out, "clustring off:  %.2lf\t%.2lf\n", 
               static_cast<double>( l_count * 100 ) / labeled.size(), 
               static_cast<double>( u_count * 100 ) / unlabeled.size() );
    }
    

    TMeans::Options clusteringOptions;
    TMeans::Clustering( box.dataset.feat, training, box.dataset.dim,
                        m_to_l, clusteringOptions );
    
    box.initVoters( inverseMap, m_to_l, numL );
    solve( numL, numU, &P[0], &m_to_l, &box.q[0] );
    
    {
      printf( "========== normal level %d ==========\n", level );
      int l_count = box.test( labeled, level );
      int u_count = box.test( unlabeled, level );

      Info( "all: %d/%ld (%.2lf)", l_count, labeled.size(), static_cast<double>( l_count * 100 ) / labeled.size() );
      Info( "all: %d/%ld (%.2lf)", u_count, unlabeled.size(), static_cast<double>( u_count * 100 ) / unlabeled.size() );
      fprintf( out, "clustering on:   %.2lf\t%.2lf\n", 
               static_cast<double>( l_count * 100 ) / labeled.size(),
               static_cast<double>( u_count * 100 ) / unlabeled.size() );
    }
    
  }
  END_WITH( out );
  return 0;
}
