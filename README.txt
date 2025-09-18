TVAUTO.PY : SCRIPT TO PROCESS TELEVIEWER WELLCAD FILES. 
--------------------------------------------------------

CHANGELOG : 
--------------------------------------------------------

DATE : 09 Sept 2025
FILE : tvauto_v2.py
USER : wai.yong
COMMENTS : 
- Filtered azi in GUI is not consistent with what's applied in rotation. 
- Changed azidf to azidf2 in def accept(self) function to fix.

DATE : 09 Sept 2025
FILE : tvauto_v3.py
USER : wai.yong
COMMENTS : 
- Added option for user to choose between DBSCAN (incumbent), LOWESS, and Alpha Trim.
- Added cascaded DBSCAN + Alpha Trim option.
- GUI updates for user friendliness.

DATE : 18 Sept 2025
FILE : tvauto_v4.py
USER : wai.yong
COMMENTS : 
- user chooses input files and output directory via GUI.
- put more details in SUMMARY console printout once done, for user to ID problems.
