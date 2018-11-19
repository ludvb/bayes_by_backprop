#!/usr/bin/env stack
{- stack
  script
  --resolver lts-12.13
  --package bytestring
  --package hashmap
  --package regex-pcre
  --package zlib
-}

import           Codec.Compression.GZip     (compress, decompress)

import           Data.ByteString.Lazy       (readFile)
import           Data.ByteString.Lazy.Char8 (pack, putStr, unpack)
import           Prelude                    hiding (putStr, readFile)

import           Data.HashSet               (fromList, member)

import           Text.Printf                (printf)

import           Text.Regex.PCRE            ((=~))

splitOn _ []        = ([], [])
splitOn c ys@(x:xs) | c == x    = ([], ys)
                    | otherwise = let (pre, post) = splitOn c xs in (x : pre, post)

parseFasta []     = []
parseFasta (x:xs) | x /= '>'  = parseFasta xs
                  | otherwise = (id, filter (/= '\n') seq) : parseFasta ys
                  where (id, rest) = splitOn '\n' xs
                        (seq, ys)  = splitOn '>' rest

getEntry :: String -> String
getEntry = (!! 1) . head . (=~ "^sp\\|(.+?)\\|")

main = do
  signalContents <- let file = "signal_prots.fasta.gz" in decompress <$> readFile file
  allContents    <- let file = "all_prots.fasta.gz"    in decompress <$> readFile file

  let signalEntries = fromList $ getEntry . fst <$> parseFasta (unpack signalContents)
  let allSeqs = parseFasta (unpack allContents)

  putStr . compress . pack . foldr1 (++) $
      printf "%s\t%s\t%s\n" "entry" "signal" "sequence" :
      flip map allSeqs ( \(id, sequence) -> do
        let entry = getEntry id
        printf "%s\t%d\t%s\n"
              entry
              (if entry `member` signalEntries then 1 else 0 :: Int)
              sequence)
