TVAUTO.PY : SCRIPT TO PROCESS TELEVIEWER WELLCAD FILES. 
--------------------------------------------------------

CHANGELOG : 
--------------------------------------------------------
09 Sept 2025 : tvauto_v2.py --> Filtered azi in GUI is not consistent with what's applied in rotation. 
Changed azidf to azidf2 in def accept(self) function to fix.

12 Sept 2025 : tvauto_v3.py --> Added option for user to choose between DBSCAN (incumbent), LOWESS, and Alpha Trim.
Added cascaded DBSCAN + Alpha Trim option.
GUI updates for user friendliness.
