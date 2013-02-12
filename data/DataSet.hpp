#pragma once

#include <type_traits>
#include <string>
#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/candy.hpp"

namespace cat_tree
{
  template <typename dataType>
  class DataSet
  {
  public:
    std::vector<std::vector<dataType> > feat;
    std::vector<int> label;
    int dim;

  public:

    // default constructor
    DataSet() : feat(), label(), dim(-1) {}
    
    DataSet( std::string filename ) 
    {
      WITH_OPEN( in, filename.c_str(), "r" );
      int len = 0;
      fread( &len, sizeof(int), 1, in );
      fread( &dim, sizeof(int), 1, in );
      feat.resize( len );
      label.resize( len );
      for ( int i=0; i<len; i++ ) {
        feat[i].resize( dim );
        fread( &feat[i][0], sizeof(dataType), dim, in );
        fread( &label[i], sizeof(int), 1, in );
        if ( 0 == i % 100 || i == len - 1 ) {
          progress( i + 1, len, "loading features" );
        }
      }
      printf( "\n" );
      END_WITH( in );
    }

    void push( std::vector<dataType>& featVec, int labelID ) 
    {
      if ( -1 == dim ) {
        dim = static_cast<int>( featVec.size() );
      } else if ( static_cast<int>( featVec.size() ) != dim ) {
        Error( "dimension mismatch, should be %d", dim );
        exit( -1 );
      }
      feat.emplace( feat.end() );
      feat.back().swap( featVec );
      label.push_back( labelID );
    }

    void write( std::string filename )
    {
      WITH_OPEN( out, filename.c_str(), "w" );
      int len = size();
      fwrite( &len, sizeof(int), 1, out );
      fwrite( &dim, sizeof(int), 1, out );
      for ( int i=0; i<len; i++ ) {
        fwrite( &feat[i][0], sizeof(dataType), dim, out );
        fwrite( &label[i], sizeof(int), 1, out );
      }
      END_WITH( out );
    }
    
    /* ---------- accessor/properties ---------- */
    inline int size() const
    {
      return static_cast<int>( feat.size() );
    }
    
  };
}
