"""What the classes this library writes look like, harvested rather than typed.

Every entry here was read out of the streamer information ROOT itself wrote
into the files under ``tests/data`` (``gauss-h1.root``, ``gauss-h2.root`` and
``graphs.root``, all written by ROOT 6.08), by ``read_streamers`` - the same
code that reads any other file. Writing carries these descriptions back out
verbatim, checksums and all, so a file written here says exactly what a ROOT
of that vintage would say about the same classes, and anything reading it -
ROOT, this library, or another - walks the bytes by the same map they were
written from.

Regenerate with ``gen_winfo.py`` against the donor files rather than editing:
a hand-edited layout is a guess, and a guess about layout is how physics data
gets misread.
"""

from __future__ import annotations

__all__ = ["INFOS", "SUBVERSIONS", "WRITER_VERSION"]

#: The ROOT version the donor descriptions came from, written into the header.
WRITER_VERSION = 60806

#: The record version each ``TStreamerElement`` subclass wrote itself with.
SUBVERSIONS = {
    "TStreamerBase": 3,
    "TStreamerBasicType": 2,
    "TStreamerBasicPointer": 2,
    "TStreamerString": 2,
    "TStreamerObject": 2,
    "TStreamerObjectAny": 2,
    "TStreamerObjectPointer": 2,
}

#: One element of one class: its ``TStreamerElement`` subclass, name, title,
#: streamer type, size, array length, array dimensions, the five maximum
#: indices, the type name, and whatever extra fields the subclass adds - the
#: base class version for a ``TStreamerBase``, and the counter's version, name
#: and class for a ``TStreamerBasicPointer``.
Element = tuple[str, str, str, int, int, int, int, tuple[int, ...], str, tuple[object, ...]]

#: Every class this library can write: its checksum, version and elements,
#: exactly as the donor files describe them.
INFOS: dict[str, tuple[int, int, tuple[Element, ...]]] = {
    "TH1D": (0xf03880de, 2, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 1063172259, 0, 0, 0), 'BASE', (7,)),
        ('TStreamerBase', 'TArrayD', 'Array of doubles',
         0, 0, 0, 0, (0, 1899622196, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH1F": (0xd91ac083, 2, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 1063172259, 0, 0, 0), 'BASE', (7,)),
        ('TStreamerBase', 'TArrayF', 'Array of floats',
         0, 0, 0, 0, (0, 1510733553, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH2D": (0xc9d05875, 3, (
        ('TStreamerBase', 'TH2', '2-Dim histogram base class',
         0, 0, 0, 0, (0, -1959965212, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerBase', 'TArrayD', 'Array of doubles',
         0, 0, 0, 0, (0, 1899622196, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH2F": (0xb2b2981a, 3, (
        ('TStreamerBase', 'TH2', '2-Dim histogram base class',
         0, 0, 0, 0, (0, -1959965212, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerBase', 'TArrayF', 'Array of floats',
         0, 0, 0, 0, (0, 1510733553, 0, 0, 0), 'BASE', (1,)),
    )),
    "TH2": (0x8b2d4de4, 4, (
        ('TStreamerBase', 'TH1', '1-Dim histogram base class',
         0, 0, 0, 0, (0, 1063172259, 0, 0, 0), 'BASE', (7,)),
        ('TStreamerBasicType', 'fScalefactor', 'Scale factor',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwy', 'Total Sum of weight*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwy2', 'Total Sum of weight*Y*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwxy', 'Total Sum of weight*X*Y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TH1": (0x3f5eb8a3, 7, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttFill', 'Fill area attributes',
         0, 0, 0, 0, (0, -2545006, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttMarker', 'Marker attributes',
         0, 0, 0, 0, (0, 689802220, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fNcells', 'number of bins(1D), cells (2D) +U/Overflows',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerObject', 'fXaxis', 'X axis descriptor',
         61, 216, 0, 0, (0, 0, 0, 0, 0), 'TAxis', ()),
        ('TStreamerObject', 'fYaxis', 'Y axis descriptor',
         61, 216, 0, 0, (0, 0, 0, 0, 0), 'TAxis', ()),
        ('TStreamerObject', 'fZaxis', 'Z axis descriptor',
         61, 216, 0, 0, (0, 0, 0, 0, 0), 'TAxis', ()),
        ('TStreamerBasicType', 'fBarOffset', '(1000*offset) for bar charts or legos',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fBarWidth', '(1000*width) for bar charts or legos',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fEntries', 'Number of entries',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumw', 'Total Sum of weights',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumw2', 'Total Sum of squares of weights',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwx', 'Total Sum of weight*X',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fTsumwx2', 'Total Sum of weight*X*X',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value for plotting',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value for plotting',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fNormFactor', 'Normalization factor',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerObjectAny', 'fContour', 'Array to display contour levels',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerObjectAny', 'fSumw2', 'Array of sum of squares of weights',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerString', 'fOption', 'histogram options',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerObjectPointer', 'fFunctions', '->Pointer to list of functions (fits and user)',
         63, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerBasicType', 'fBufferSize', 'fBuffer size',
         6, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicPointer', 'fBuffer', '[fBufferSize] entry buffer',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (7, 'fBufferSize', 'TH1')),
        ('TStreamerBasicType', 'fBinStatErrOpt', 'option for bin statistical errors',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'TH1::EBinErrorOpt', ()),
    )),
    "TGraph": (0x05f7f465, 4, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttLine', 'Line attributes',
         0, 0, 0, 0, (0, -1811462839, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttFill', 'Fill area attributes',
         0, 0, 0, 0, (0, -2545006, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBase', 'TAttMarker', 'Marker attributes',
         0, 0, 0, 0, (0, 689802220, 0, 0, 0), 'BASE', (2,)),
        ('TStreamerBasicType', 'fNpoints', 'Number of points <= fMaxSize',
         6, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicPointer', 'fX', '[fNpoints] array of X points',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fY', '[fNpoints] array of Y points',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerObjectPointer', 'fFunctions', 'Pointer to list of functions (fits and user)',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
        ('TStreamerObjectPointer', 'fHistogram', 'Pointer to histogram used for drawing axis',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TH1F*', ()),
        ('TStreamerBasicType', 'fMinimum', 'Minimum value for plotting along y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fMaximum', 'Maximum value for plotting along y',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
    )),
    "TGraphErrors": (0x2a7ce30f, 3, (
        ('TStreamerBase', 'TGraph', 'Graph graphics class',
         0, 0, 0, 0, (0, 100136037, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerBasicPointer', 'fEX', '[fNpoints] array of X errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fEY', '[fNpoints] array of Y errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
    )),
    "TGraphAsymmErrors": (0xcc46af3b, 3, (
        ('TStreamerBase', 'TGraph', 'Graph graphics class',
         0, 0, 0, 0, (0, 100136037, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerBasicPointer', 'fEXlow', '[fNpoints] array of X low errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fEXhigh', '[fNpoints] array of X high errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fEYlow', '[fNpoints] array of Y low errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
        ('TStreamerBasicPointer', 'fEYhigh', '[fNpoints] array of Y high errors',
         48, 8, 0, 0, (0, 0, 0, 0, 0), 'double*', (4, 'fNpoints', 'TGraph')),
    )),
    "TNamed": (0xdfb74a3c, 1, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerString', 'fName', 'object identifier',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerString', 'fTitle', 'object title',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
    )),
    "TObject": (0x901bc02d, 1, (
        ('TStreamerBasicType', 'fUniqueID', 'object unique identifier',
         13, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
        ('TStreamerBasicType', 'fBits', 'bit field status word',
         15, 4, 0, 0, (0, 0, 0, 0, 0), 'unsigned int', ()),
    )),
    "TAttLine": (0x94074549, 2, (
        ('TStreamerBasicType', 'fLineColor', 'Line color',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLineStyle', 'Line style',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLineWidth', 'Line width',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
    )),
    "TAttFill": (0xffd92a92, 2, (
        ('TStreamerBasicType', 'fFillColor', 'Fill area color',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fFillStyle', 'Fill area style',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
    )),
    "TAttMarker": (0x291d8bec, 2, (
        ('TStreamerBasicType', 'fMarkerColor', 'Marker color',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fMarkerStyle', 'Marker style',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fMarkerSize', 'Marker size',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
    )),
    "TAxis": (0x5a496e70, 10, (
        ('TStreamerBase', 'TNamed', 'The basis for a named object (name, title)',
         67, 0, 0, 0, (0, -541636036, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerBase', 'TAttAxis', 'Axis attributes',
         0, 0, 0, 0, (0, 1550843710, 0, 0, 0), 'BASE', (4,)),
        ('TStreamerBasicType', 'fNbins', 'Number of bins',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fXmin', 'low edge of first bin',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerBasicType', 'fXmax', 'upper edge of last bin',
         8, 8, 0, 0, (0, 0, 0, 0, 0), 'double', ()),
        ('TStreamerObjectAny', 'fXbins', 'Bin edges array in X',
         62, 24, 0, 0, (0, 0, 0, 0, 0), 'TArrayD', ()),
        ('TStreamerBasicType', 'fFirst', 'first bin to display',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fLast', 'last bin to display',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fBits2', 'second bit status word',
         12, 2, 0, 0, (0, 0, 0, 0, 0), 'unsigned short', ()),
        ('TStreamerBasicType', 'fTimeDisplay', 'on/off displaying time values instead of numerics',
         18, 1, 0, 0, (0, 0, 0, 0, 0), 'bool', ()),
        ('TStreamerString', 'fTimeFormat', 'Date&time format, ex: 09/12/99 12:34:00',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerObjectPointer', 'fLabels', 'List of labels',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'THashList*', ()),
        ('TStreamerObjectPointer', 'fModLabs', 'List of modified labels',
         64, 8, 0, 0, (0, 0, 0, 0, 0), 'TList*', ()),
    )),
    "TAttAxis": (0x5c6fff3e, 4, (
        ('TStreamerBasicType', 'fNdivisions', 'Number of divisions(10000*n3 + 100*n2 + n1)',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
        ('TStreamerBasicType', 'fAxisColor', 'Color of the line axis',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLabelColor', 'Color of labels',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLabelFont', 'Font for labels',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fLabelOffset', 'Offset of labels',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fLabelSize', 'Size of labels',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTickLength', 'Length of tick marks',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTitleOffset', 'Offset of axis title',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTitleSize', 'Size of axis title',
         5, 4, 0, 0, (0, 0, 0, 0, 0), 'float', ()),
        ('TStreamerBasicType', 'fTitleColor', 'Color of axis title',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
        ('TStreamerBasicType', 'fTitleFont', 'Font for axis title',
         2, 2, 0, 0, (0, 0, 0, 0, 0), 'short', ()),
    )),
    "THashList": (0xcc7e49c1, 0, (
        ('TStreamerBase', 'TList', 'Doubly linked list',
         0, 0, 0, 0, (0, 1774568379, 0, 0, 0), 'BASE', (5,)),
    )),
    "TList": (0x69c5c3bb, 5, (
        ('TStreamerBase', 'TSeqCollection', 'Sequenceable collection ABC',
         0, 0, 0, 0, (0, -60015674, 0, 0, 0), 'BASE', (0,)),
    )),
    "TSeqCollection": (0xfc6c3bc6, 0, (
        ('TStreamerBase', 'TCollection', 'Collection abstract base class',
         0, 0, 0, 0, (0, 1474546588, 0, 0, 0), 'BASE', (3,)),
    )),
    "TCollection": (0x57e3cb9c, 3, (
        ('TStreamerBase', 'TObject', 'Basic ROOT object',
         66, 0, 0, 0, (0, -1877229523, 0, 0, 0), 'BASE', (1,)),
        ('TStreamerString', 'fName', 'name of the collection',
         65, 24, 0, 0, (0, 0, 0, 0, 0), 'TString', ()),
        ('TStreamerBasicType', 'fSize', 'number of elements in collection',
         3, 4, 0, 0, (0, 0, 0, 0, 0), 'int', ()),
    )),
    "TString": (0x00017419, 2, (
    )),
}
