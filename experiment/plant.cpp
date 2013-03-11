#include <vector>
#include <string>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/Environment.hpp"
#include "../data/Label.hpp"
#include "../data/DataSet.hpp"
#include "RanForest/RanForest.hpp"

using namespace EnvironmentVariable;
using namespace ran_forest;
using namespace cat_tree;

int main( int argc, char **argv )
{
  if ( argc < 2 ) {
    Error( "Missing configuration file in options. (maybe plant.conf?)" );
    exit( -1 );
  }
  
  env.parse( argv[1] );
  env.Summary();
  
  LabelSet::initialize( env["class-map"] );
  LabelSet::Summary();

  DataSet<float> dataset( env["feature-data-input"] );
  



  if ( true ) {
    typename SimpleKernel<std::vector<float>, BinaryOnAxis>::Options options;
    options.dim = dataset.dim;
    options.converge = 0.01;
    options.stopNum = 5;
  
    Forest<float,BinaryOnAxis> forest;
    forest.grow<SimpleKernel>( env["forest-size"],
                               dataset.feat,
                               options,
                               env["propotion-per-tree"].toDouble() );
  
    printf( "nodeNum: %d\n", forest.nodeNum() );
    printf( "leafNum: %d\n", forest.levelSize(100) );

    forest.write( env["forest-output-dir"] );
    Done( "forest written to %s", env["forest-output-dir"].c_str() );
  } else {
    typename MaxGapSubspaceKernel<std::vector<float>, BinaryOnSubspace>::Options options;
    options.dim = dataset.dim;
    options.converge = 0.01;
    options.stopNum = 5;
    options.dimPrelim = 40;
    options.dimFinal = 100;
    options.numHypo = 10;
    
  
    Forest<float,BinaryOnSubspace> forest;
    forest.grow<MaxGapSubspaceKernel>( env["forest-size"],
                                       dataset.feat,
                                       options,
                                       env["propotion-per-tree"].toDouble() );
  
    printf( "nodeNum: %d\n", forest.nodeNum() );
    printf( "leafNum: %d\n", forest.levelSize(100) );

    forest.write( env["forest-output-dir"] );
    Done( "forest written to %s", env["forest-output-dir"].c_str() );

  }
  return 0;
}

