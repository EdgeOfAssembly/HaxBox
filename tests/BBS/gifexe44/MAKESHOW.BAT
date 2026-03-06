rem
rem First put a directory of all the gifs in the current directory
rem into a file...
rem 
dir /b *.gif > list
rem
rem Now all the GIF files will be in the file "list"
rem
rem Edit the file "list" and add any options that you want to each GIF
rem in the list.
rem
rem Example:
rem
rem If the file "list" looks like:
rem
rem          sunset.gif
rem          bike.gif
rem          mountain.gif
rem
rem Then edit in the (registered version) options that you want for each GIF:
rem
rem          sunset.gif  /fadein /delay=2
rem          bike.gif /delay=2
rem          mountain.gif /delay=2 /fadeout
rem
rem If you don't edit the file to add options like /delay and /fadeout,
rem then a "standard" slideshow program will be produced, and you will have to 
rem press a key after each GIF to go on to the next GIF.
rem
rem To make the slideshow program from your file "list", use the following
rem command:
rem
rem gifexe @list
rem
rem or
rem
rem gifexe @list /cycle=1
rem
rem for a continuous slideshow.
rem

