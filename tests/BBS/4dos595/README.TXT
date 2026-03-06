                               JP Software

                               4DOS v. 5.95

                  Pre-Release README.TXT -- May 27, 1997


Greetings, and thanks for trying this pre-release!

   PLEASE NOTE:  This file accompanies a pre-release software product
   (4DOS 5.95, 4NT 2.95, Take Command/16 1.95, or Take Command/32 1.95).
   These versions EXPIRE on July 31, 1997, as does your license to use
   them (for the full license, see LICENSE.TXT).  The software will warn
   you when expiration is getting close; once the pre-release has
   expired you will be subject to reminders and significant startup
   delays each time you use it.  You will be able to download a fully
   released copy (or an updated pre-release version) before expiration
   of this pre-release.

This file contains a variety of information you should read before using
the pre-release, including:

     Files Included with the Pre-Release
     Installation Instructions
     Technical Support
     Ordering the Final Release
     Legal Notices


FILES INCLUDED
--------------

   The following files are included with 4DOS 5.95:

        4DOS.COM            4DOS program file
        4DOS.HLP            4DOS help text
        4HELP.EXE           4DOS help program
        FILE_ID.DIZ         Description file for BBS and other uploads
        HELPCFG.EXE         Color configuration program for the help
                              system
        INTERNAT.TXT        Shipping rates and dealer list for
                              international customers
        KSTACK.COM          Memory-resident program used by the KEYSTACK
                              command
        LICENSE.TXT         Pre-release license agreement
        OPTION.EXE          Program used by the OPTION command
        ORDFORM.TXT         New customer order form
        PRODUCTS.TXT        Product and ordering information
        README.TXT          This file
        UPGFORM.TXT         New customer order form


INSTALLATION INSTRUCTIONS
-------------------------
   [NOTE to current JP Software command processor users:  If you are
   running an earlier version of our software you may be used to
   "branding" new versions with your name and serial number.  However,
   you can NOT brand this pre-release software; as a pre-release, it
   cannot be used permanently, and therefore does not accept branding.
   Your brand program will still work on the final release if you are
   entitled to upgrade to it at no charge.]


   Here's how to install your pre-release software:

   (1) Create a new directory for the pre-release software.  Do not
       install it in the same directory as any other JP Software
       product, including one of the other pre-release products.  The
       directory can be on any hard disk drive.

       When installing 4DOS under Windows 95 we recommend that you do NOT
       use a "long" directory name.  If you do, you will have to know and
       use the equivalent short name to load 4DOS if the Windows 95 GUI is
       not loaded (e.g., when loading 4DOS in CONFIG.SYS).

   (2) Decompress the downloaded files into this directory.  You can print
       any documentation (.TXT) file by copying it to the printer with the
       COPY command.

   (3) Add this line to your DOS, Windows 95, or OS/2 CONFIG.SYS file:
            
            SHELL=d:\path\4DOS.COM d:\path /P
       
       "d:\path" means the drive and directory where 4DOS.COM is stored.
       The second "d:\path" on the SHELL= line should be the same as the
       first, and is used to set the COMSPEC environment variable
       properly.  Be sure to delete or REMark out any old SHELL= line
       for COMMAND.COM (or a previous version of 4DOS) after you add the
       new SHELL= line.
       
       You can add any command line options to the end of the line.  See
       the online help for information on command line options.
       
       If you are running 4DOS under DOS 4.01 or earlier, you must make
       sure that the portion of the SHELL= line beginning with the
       second "d:\path", through the end of the line, does not exceed 31
       characters, due to a bug in DOS.  For details, see the
       Compatibility section in the online help.
       
   (4) If you are running 4DOS under OS/2, change CONFIG.SYS as
       described above (don't forget to find and REMark out or delete
       the old SHELL= line).  Once you reboot, 4DOS will be used
       automatically for all DOS objects which use the default DOS_SHELL
       settings.
       
   (5) Next, if you are running under DOS, OS/2, or Windows 3.x, add the
       following line to your AUTOEXEC.BAT file:
            
            d:\path\KSTACK.COM
       
       where "d:\path" is the drive and directory where your 4DOS files
       are stored.  KSTACK.COM is a memory-resident program used to
       support the KEYSTACK command.  It requires about 1K of memory,
       and can be left out of AUTOEXEC.BAT if you do not use KEYSTACK.
       
       Do not add the KSTACK line to AUTOEXEC.BAT if you are using
       Windows 95.  Under Windows 95, KSTACK should be loaded separately
       for each 4DOS session; see the Windows 95 information in the
       Compatibility section of the online help for details.
       
       If you have a line in AUTOEXEC.BAT which sets the COMSPEC
       environment variable to point to COMMAND.COM, remove it or REMark
       it out.  If you set up the CONFIG.SYS file properly, 4DOS will
       set this variable automatically; setting it in AUTOEXEC.BAT is
       likely to cause trouble later on (for example, if you install an
       update to 4DOS in a different directory, but forget to change the
       COMSPEC setting).
       
   (6) 4DOS requires a 4DOS.INI file with at least two lines in it:
            
            [4DOS]
            InstallPath = d:\path
       
       where "d:\path" is the drive and directory where your 4DOS files
       are stored.  This will help 4DOS locate its auxiliary files, such
       as the help files and OPTION.EXE.  The [4DOS] section header
       should be the first line in the file, and the InstallPath setting
       should be the second line.
       
       If you are upgrading from a previous version of 4DOS, you should
       copy any 4DOS.INI file you have in the old directory to the new
       one.  Then add or modify the [4DOS] section head and InstallPath
       directive at the beginning of 4DOS.INI, remove any HelpPath
       directive in the file (HelpPath is obsolete and has been replaced
       by InstallPath), and update any paths in the file which point to
       the old program directory (e.g. 4StartPath).

   (7) If you are updating from a previous version of 4DOS you should
       also copy any 4START or 4EXIT files kept in your old 4DOS
       directory to the new directory.

   (8) When you've finished modifying CONFIG.SYS and AUTOEXEC.BAT, and
       updating your other files, you can reboot your system to start
       4DOS.

   (9) If you are running Windows 95, you will need to create a new
       shortcut on your desktop to run 4DOS (the "MS-DOS Prompt"
       shortcut which comes with Windows 95 will run COMMAND.COM, even
       after 4DOS is installed).  Simply create a new shortcut as you
       would for any program (right-click on the open desktop, click
       New, then Shortcut), and enter the Command Line as:
       
            d:\path\4dos.com d:\path

       where "d:\path" is the directory where your 4DOS files are stored
       [this is the same entry as described for the SHELL= line in step
       (3) above, but without the final "/P"].

       Once you have 4DOS running, be sure to look through the material
       on Windows 95 in the Compatibility section of the online help.
       It explains in detail how to configure 4DOS and Windows 95 so
       that they work well together.


TECHNICAL SUPPORT
-----------------
   We support our products via two electronic methods:

      * In the JP Software CompuServe area, section 10 of the PC Vendor
        B forum (GO PCVENB).  For support leave a message in this area
        addressed to user ID 75300,1215.
   
      * On the Internet, via email to support@jpsoft.com.

   When possible, we also read messages posted on the Usenet
   comp.os.msdos.4dos newsgroup, and in the 4DOS conference on the
   FidoNet BBS network (these conferences are named for 4DOS, but carry
   messages related to all JP Software products).  These areas offer
   valuable information and discussions with other users, but are not
   managed by JP Software, and are not official support channels.  To be
   certain of a direct answer from our support staff use our CompuServe
   forum or Internet email as listed above.

   Because this is a pre-release version, NO TELEPHONE SUPPORT IS
   AVAILABLE.  Limited electronic support is available through
   CompuServe or the Internet as noted above, and support questions may
   be answered from time to time in the newsgroup or BBS conferences
   listed above.  However, this is a pre-release product and may contain
   bugs or other limitations.  We cannot promise specific answers to all
   problems encountered with this version, offer a definite response
   time for questions on the pre-release, or provide projected dates for
   updates or fixes.


   Downloading JP Software Files

      To download JP Software files, including a trial version of
      the final release of this software (when it is available), visit:

         * Our web site at http://www.jpsoft.com/

         * Our CompuServe forum (GO PCVENB or GO JPSOFT, library 10 / JP
           Software)

         * Channel 1 BBS, Cambridge, MA, 617-349-1300 for 28.8K, 617-
           354-3137 for 33.6 K. Join the "free files" conference (J FREE
           or J 5 at the main menu) for JP Software file downloads.

      We recommend that you use our web site for Internet downloads.
      However, if you prefer (or if you do not have web access) you can
      connect directly to our ftp site at ftp.std.com, directory
      /vendors/jpsoft.  Product subdirectories are located below this;
      for example 4DOS files are in /vendors/jpsoft/4dos, and
      pre-release files are in /vendors/jpsoft/prerel.


ORDERING THE FINAL RELEASE
--------------------------
   No date is currently available for the final release.

   Current users can place advance upgrade orders using the upgrade
   order form distributed with the pre-release, or using electronic
   order entry via our web site at http://www.jpsoft.com/ (web-based
   order entry system is not available at this time, but will be ready
   shortly).

   All current users will also be notified by mail when the final
   release is available.

   Orders from new customers, and add-on orders from existing customers,
   which are placed prior to the final release of this upgrade will be
   filled immediately with our current version of the product(s)
   ordered.  The new version will then be shipped automatically when it
   is ready.

   Single-product upgrades for owners of 4DOS 5.0 or above, 4OS2 or 4NT
   2.0 or above, or any version of Take Command will be priced at
   $29.95 for the first product and $19.95 for each additional product,
   plus shipping, through August 31, 1997, and will increase after that
   date.  (Upgrades from earlier versions -- 4DOS 4.02 or below, 4OS2 or
   4NT 1.x -- are $39.95 plus shipping, also through August 31.)

   We are also offering attractive pricing for those who want to convert
   to the JP CD Suite with all six products -- $89.95 for owners of one
   product, and just $69.95 for those who own two or more JP Software
   products -- this means if you are upgrading three or more products
   you can convert to the CD Suite for the same amount you'd pay for
   your upgrade, or less!

   If you already own the CD Suite, upgrades are $49.95 if you have CD
   Release 2 with all six products, and $59.95 if you have Release 1
   with 5 products.


LEGAL STUFF
-----------

Copyright 1997, JP Software Inc., All Rights Reserved.  Published by JP
Software Inc., P.O. Box 1470, E. Arlington, MA 02174, U.S.A., phone
617-646-3975, fax 617-646-0904.

Take Command and 4DOS are registered trademarks of JP Software Inc.
4DOS, 4OS2, and 4NT are JP Software Inc.'s trademarks for its family of
character-mode command processors.  JP Software, jpsoft.com, and all JP
Software designs and logos are also trademarks of JP Software Inc. Other
product and company names are trademarks of their respective owners.

All prices and terms subject to change without notice.

This pre-release version may be incomplete, and may contain errors or
omissions which would not be present in a final release product.  It is
distributed AS-IS and without warranty of any kind.  See the license
agreement accompanying the pre-release files (in the file LICENSE.TXT)
for details.

