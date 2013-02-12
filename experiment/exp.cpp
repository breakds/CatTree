#include <vector>
#include <string>
#include <unordered_set>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "LLPack/algorithms/random.hpp"
#include "../data/Label.hpp"
#include "../data/DataSet.hpp"
#include "../data/Bipartite.hpp"
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

  /* loading dataset */
  DataSet<float> dataset( env["feature-data-input"] );
  
  
  /* loading forest */
  Forest<float,BinaryOnAxis> forest( env["forest-dir"] );
  printf( "nodeNum: %d\n", forest.nodeNum() );

  
  /* generate random partition */
  std::vector<int> testing = rndgen::seq( dataset.size() );
  std::vector<int> labeled;
  std::vector<int> unlabeled;
  partition( dataset, testing, labeled, env["testing-ratio"].toDouble() );
  partition( dataset, labeled, unlabeled, env["labeled-ratio"].toDouble() );
  Info( "labeled: %ld", labeled.size() );
  Info( "unlabeled: %ld", unlabeled.size() );
  Info( "testing: %ld", testing.size() );

  /* creating index inverse map */
  std::vector<int> inverseMap( dataset.size() );
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
  
  /* load patch pairs */
  std::vector<std::pair<int,int> > patchPairs;
  std::vector<double> w;
  WITH_OPEN( in, env["knn-input"].c_str(), "r" );
  int n = 0;
  fread( &n, sizeof(int), 1, in );
  patchPairs.reserve( n );
  w.reserve( n );
  int a = 0;
  int b = 0;
  double dist = 0.0;
  double sigma = env["sigma"].toDouble();
  for ( int i=0; i<n; i++ ) {
    fread( &a, sizeof(int), 1, in );
    fread( &b, sizeof(int), 1, in );
    fread( &dist, sizeof(double), 1, in );
    if ( -1 == inverseMap[a] || -1 == inverseMap[b] ) {
      continue;
    }
    patchPairs.push_back( std::make_pair( a, b ) );
    w.push_back( ExpConv( dist, sigma ) );
  }
  END_WITH( in );





  
  
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

  
  
  /* forest query */
  Bipartite m_to_l( M, forest.nodeNum() );
  std::vector<double> P( numL * LabelSet::classes );
  memset( &P[0], 0, sizeof(double) * numL * LabelSet::classes );
  {
    int m = 0;
    for ( auto& ele : labeled ) {
      auto res = forest.query( dataset.feat[ele], 4 );
      double alpha = 1.0 / res.size();
      for ( auto& item : res ) {
        m_to_l.add( m, item, alpha );
      }
      
      P[ m * LabelSet::classes + dataset.label[ele] ] = 1.0;
      m++;
      if ( 0 == m % 100 ) {
        progress( m, labeled.size() + unlabeled.size(), "query" );
      }
    }

    for ( auto& ele : unlabeled ) {
      auto res = forest.query( dataset.feat[ele], 4 );
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
  
  /* pair_to_l */
  Bipartite pair_to_l( static_cast<int>( patchPairs.size() ), forest.nodeNum() );
  {
    int n = 0;
    for ( auto& ele : patchPairs ) {
      std::unordered_set<int> hash;
      {
        int i = inverseMap[ele.first];
        auto& _to_l = m_to_l.getToSet( i );
        for ( auto& item : _to_l ) {
          if ( hash.end() == hash.find( item.first ) ) {
            pair_to_l.add( n, item.first, 1.0 );
            hash.insert( item.first );
          }
        }
      }

      {
        int j = inverseMap[ele.second];
        auto& _to_l = m_to_l.getToSet( j );
        for ( auto& item : _to_l ) {
          if ( hash.end() == hash.find( item.first ) ) {
            pair_to_l.add( n, item.first, 1.0 );
            hash.insert( item.first );
          }
        }
      }
      n++;
      if ( 0 == n % 100 ) {
        progress( n, patchPairs.size(), "build pair_to_l" );
      }
    }
    progress( 1, 1, "build pair_to_l" );
    printf( "\n" );
  }

  /* solve */
  Solver solve;
  solve.options.beta = env["beta"].toDouble();
  solve.options.maxIter = env["max-iter"];
  

  std::vector<double> q( forest.nodeNum() * LabelSet::classes );
  for ( int i=0; i<forest.nodeNum() * LabelSet::classes; i++ ) {
    q[i] = LabelSet::inv;
  }


  Info( "solving..." );
  solve( numL, numU, forest.nodeNum(),
         &m_to_l, &pair_to_l, &patchPairs,
         &w[0], &P[0], &q[0] );
  Done( "Solved" );


  /* Voting */
  {
    double vote[LabelSet::classes];
    int l_count = 0;
    for ( auto& ele : labeled ) {
      auto res = forest.query( dataset.feat[ele], 4 );
      memset( vote, 0, sizeof(double) * LabelSet::classes );
      for ( auto& item : res ) {
        addto( vote, &q[item * LabelSet::classes], LabelSet::classes );
      }
      for ( int i=0; i<LabelSet::classes; i++ ) {
        vote[i] *= LabelSet::GetWeight(i);
      }

      int infer = 0;
      for ( int i=1; i<LabelSet::classes; i++ ) {
        if ( vote[i] > vote[infer] ) infer = i;
      }
      if ( infer == dataset.label[ele] ) {
        l_count++;
      }
    }

    
    int u_count = 0;
    for ( auto& ele : unlabeled ) {
      auto res = forest.query( dataset.feat[ele], 4 );
      memset( vote, 0, sizeof(double) * LabelSet::classes );
      for ( auto& item : res ) {
        addto( vote, &q[item * LabelSet::classes], LabelSet::classes );
      }
      for ( int i=0; i<LabelSet::classes; i++ ) {
        vote[i] *= LabelSet::GetWeight(i);
      }

      int infer = 0;
      for ( int i=1; i<LabelSet::classes; i++ ) {
        if ( vote[i] > vote[infer] ) infer = i;
      }
      if ( infer == dataset.label[ele] ) {
        u_count++;
      }
    }

        
    int t_count = 0;
    for ( auto& ele : testing ) {
      auto res = forest.query( dataset.feat[ele], 4 );
      memset( vote, 0, sizeof(double) * LabelSet::classes );
      for ( auto& item : res ) {
        addto( vote, &q[item * LabelSet::classes], LabelSet::classes );
      }
      for ( int i=0; i<LabelSet::classes; i++ ) {
        vote[i] *= LabelSet::GetWeight(i);
      }

      int infer = 0;
      for ( int i=1; i<LabelSet::classes; i++ ) {
        if ( vote[i] > vote[infer] ) infer = i;
      }
      if ( infer == dataset.label[ele] ) {
        t_count++;
      }
    }

    Info( "labeled: %d/%ld (%.2lf)", l_count, labeled.size(), static_cast<double>( l_count * 100 ) / labeled.size() );
    Info( "unlabeled: %d/%ld (%.2lf)", l_count, unlabeled.size(), static_cast<double>( l_count * 100 ) / unlabeled.size() );
    Info( "testing: %d/%ld (%.2lf)", l_count, testing.size(), static_cast<double>( l_count * 100 ) / testing.size() );
  }
  
  
  return 0;
}
