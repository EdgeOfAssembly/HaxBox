       *************************************************************
       **************    ===RKIVE file archiver===    **************
       *************************************************************
            
            Copyright (c)1996 Malcolm Taylor All Rights Reserved


      RKIVE is a high performance shareware file archiver which uses many 
   of the latest techniques to gain a high level of compression while
   maintaining a reasonable decompression speed.

   This is version 1.4 - released on 30th October 1996.

* Features:

   RKIVE now supports a full range of standard archiver features - these
   are:

         - Support for file attributes.
         - Support for file date/time stamps.
         - Support for CRC32 checksums to verify file integrity.
         - Support for Win95 long filenames.
         - Better compression than almost any other archiver.
         - Environment variable argument passing.


* Requirements:

   This program will run on any 386 or better computer. 4Mb of memory is
   required but 16Mb is recommended for the high compression modes.


* Format and future:

   The format and compression algorithm being used by RKIVE changes
   significantly between versions as I investigate ways of improving the
   compression. Because of this, the versions are not backwardly compatible
   (an archive compressed by v1.3 will not be able to be decompressed by
   v1.4). This version will give an indication of the version required to
   decompress any incompatible archive files.

   Because of the problems that this causes I am aiming to freeze the
   algorithm and format once I've finished investigating the algorithm
   and have completed a full feature set. This will then be released as
   v2.0 of RKIVE with the stipulation that any subsequent versions will
   be backwardly compatible.


* Shareware info:

   If you find this program useful then please register it by sending
   US$20 to the address given below.


* Legal info:

   The Author takes no responsibility for any damages resulting from the
   use or misuse of this program.
   USE THIS PROGRAM AT YOUR OWN RISK.

   This software is distributed as Shareware and is Copyright (c) 1996
   Malcolm Taylor, all rights reserved.


   Having said this if you have any problems then please feel free to 
   contact me at the e-mail address below.


* Usage:

   The usage of the archiver is fairly self explanatory:

      rkive [options] filename[.rkv] [files ... @list]

   The options are:
      -c           to compress (defaults to this if no options given).
      -e           to expand without paths.
      -x           to expand with paths.
      -i           to perform an integrity check.
      -v           to view the files.
      -r           recurse through sub-directories.
      -p[0,r,f]    store pathnames:
                      0 - no pathnames.
                      r - relative pathnames (default).
                      f - full pathnames.
      -s[0,t,e,n]  sets the sorting method to:
                      0 - no sorting.
                      t - sort by file type (default).
                      e - sort by extension.
                      n - sort by name.
      -m[f,b,t[x]] sets the compression mode (see below).
      -mm[0..2]    sets the multi-media compression mode to:
                      0 - no multi-media compression.
                      1 - file-type selected multi-media compression.
                      2 - forced all files multi-media compression.
      -o[+/-]      turns the compression optimisations on or off.
      -j[rsh][+-]  includes files with given attributes.
      -tNNNN       sets the table size (sets compression level).
      -bNNNN       sets the buffer size (see below).

   After the options come the archive filename and then the list of files
   for the given operation. A file containing a list of files to compress
   can also be used by prefixing the list's filename with '@'.


* Compression modes:

   There are four main compression modes that have been designed to give
   a wide range of speed and compression. These are:

   -mf
      This mode is the fastest mode, sacrificing compression ratio to
   gain better speed (despite this the compression is often better than
   many other popular archivers).

   -mb
      This is the default mode, and is best on binary files. On all files
   this gives the best compression for a reasonable speed.

   -mt[x]
      These modes both require 16Mb of physical memory to perform at their
   top speed, otherwise they can often be significantly slower. These modes
   both give excelent text compression, however on some binary file types
   (most notably executables and object code) they perform about 1% worse
   than -mb. The -mt mode is the fastest of the two modes, providing
   almost as good compression as -mtx but taking about 30% less time.

   There are also available options to change the sizes of internal tables,
   allowing slightly better compression in some cases. In most cases
   changing these values has a slight negative effect, so these are
   normally best left alone.


* Examples of use:

      rkive archive *.txt *.exe @list

   This will compress all *.txt, *.exe and the files listed in the file
   'list' to the archive file archive.rkv using the default binary mode.

      rkive -jr+s-h -mtx archive *.txt *.doc "Win95 long name.txt"

   This will compress all *.txt, *.doc and the file 'Win95 long name.txt'
   including files with the read-only attribute and hidden attribute but
   not files with the system attribute to the archive file archive.rkv
   using the max-text mode.

      rkive -x archive

   This will extract all files in archive including any stored pathnames.

   NB. There does not need to be any files listed to extract, as in this
       case all files are extracted.


* Memory:

   Given below is a list of the approximate memory requirements for each
   mode.

      Mode         Mem for compr.     Mem for extract.
      -mf              3.3Mb              3.0Mb
      -mb              6.4Mb              5.8Mb
      -mtf            11.4Mb             11.1Mb
      -mt             13.6Mb             13.0Mb

   If you do not have this much available memory (ie. free _PHYSICAL_
   memory) then speed can be significantly hurt. The archiver will still
   function though.


* Environment variable:

   Whenever RKIVE is executed it looks for the environment variable
   'RKIVE_OPT' and parses the value of this variable as though it were
   command line arguments. This enables the default behaviour of RKIVE
   to be changed, or favourite modes to be set automatically. Any
   arguments in the environment variable can be overridden on the command
   line.

   eg. SET RKIVE_OPT=-p0 -sn -mt

   Will make RKIVE use the text mode, store no paths and sort by name
   by default when rkive is next executed.


* Contact info:

   You can contact me if you have any advice, comments or requests for
   info via the Internet at:
      malcolm@manawatu.gen.nz

   My snail mail address is:
      Malcolm Taylor
      19 Koromiko Road
      Wanganui
      New Zealand


* Acknowledgments:

   I'd like to acknowledge some people for their help in creating RKIVE.

   The most notable is Charles Bloom without whose helpful pointers,
   information and homepage I would never have been able to write RKIVE.

   Philip Lai and Joe Negron need special thanks for the generous
   donation of their time in testing pre-release versions of RKIVE and
   helping me locate bugs.

   A general thanks as well to all others who have sent me mail about
   bugs or suggestions for new features.


