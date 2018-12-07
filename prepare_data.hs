#!/usr/bin/env stack
{- stack
  script
  --resolver lts-12.13
  --package array
  --package bytestring
  --package regex-pcre
  --package text
  --package transformers
  --package unordered-containers
  --package zlib
-}

{-# LANGUAGE ExtendedDefaultRules #-}
{-# LANGUAGE OverloadedStrings    #-}

import           Control.Monad              (forM_)
import           Control.Monad.ST           (ST (..))
import           Control.Monad.Trans.Class  (lift)
import           Control.Monad.Trans.Maybe  (MaybeT (..), runMaybeT)

import           Codec.Compression.GZip     (compress, decompress)

import           Data.ByteString.Lazy       (readFile)
import           Data.ByteString.Lazy.Char8 (pack, putStr, unpack)

import           Data.Array.IArray          ((!))
import           Data.Array.ST              (STUArray (..), newArray, readArray,
                                             runSTUArray, writeArray)

import           Data.HashMap.Strict        (fromList, lookup, toList)

import qualified Data.HashSet               as Set (fromList, member)

import           Data.List                  (intercalate)
import           Data.Tuple                 (swap)

import qualified Data.Text                  as T

import           Text.Printf                (printf)

import           Text.Regex.PCRE            ((=~))

import           Prelude                    hiding (lookup, putStr, readFile)


default (String)


aminoAcids = "ACDEFGHIKLMNPQRSTVWY"

parseFasta :: String -> [(String, String)]
parseFasta (_:xs) =
  (\(a:bs) -> (T.unpack a, (T.unpack . T.concat) bs)) . T.lines
  <$> T.splitOn "\n>" (T.pack xs)

getEntry :: String -> String
getEntry = (!! 1) . head . (=~ "^sp\\|(.+?)\\|")

n `gramsOf` xs | length xs < n = []
               | otherwise     = take n xs : n `gramsOf` tail xs

n `gramDictionaryOf` xs =
  let ngrams = sequence (replicate n xs)
  in fromList (zip ngrams [1..])

bagOfWords dictionary words = runSTUArray $ do
  let n = length dictionary
  result <- newArray (1, n) 0 :: ST s (STUArray s Int Int)
  forM_ words $ \word -> runMaybeT $ do
    wordIndex <- MaybeT $ return $ word `lookup` dictionary
    currentValue <- lift $ readArray result wordIndex
    lift $ writeArray result wordIndex (currentValue + 1)
  return result

main = do
  let n = 3
  let ngram2id = n `gramDictionaryOf` aminoAcids

  signalContents <- let file = "signal_prots.fasta.gz" in decompress <$> readFile file
  allContents    <- let file = "all_prots.fasta.gz"    in decompress <$> readFile file

  let signalEntries = Set.fromList $ getEntry . fst <$> (parseFasta . unpack) signalContents
  let allSeqs = (parseFasta . unpack) allContents


  let (ngrams, ids) = foldr (\(x, y) (a, b) -> (x : a, y : b)) ([], []) (toList ngram2id)
    in putStr . compress . pack . foldr1 (++) $
      (intercalate "\t" ([ "entry" , "signal" , "sequence" ] ++ ngrams)
       ++ "\n")
      : (<$> allSeqs) (\(id, sequence) -> do
        let entry = getEntry id
        let bow = bagOfWords ngram2id (n `gramsOf` sequence)
        intercalate "\t" (
            [ entry
            , show (if entry `Set.member` signalEntries then 1 else 0 :: Int)
            , sequence
            ]
            ++ (show . (bow !) <$> ids)
          ) ++ "\n")
