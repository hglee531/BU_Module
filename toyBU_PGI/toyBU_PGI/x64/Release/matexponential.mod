V29 :0x14 matexponential
58 F:\Archive\toy\Depletion\toyBU_PGI\toyBU_PGI\MatExpMod.f90 S624 0
08/21/2018  14:36:35
use csrmatrix public 0 direct
use cudafor_lib public 0 indirect
use cudafor public 0 direct
use pgi_acc_common public 0 indirect
use cublas public 0 direct
use iso_c_binding public 0 indirect
use cusparse public 0 direct
enduse
D 56 24 647 8 646 7
D 62 24 649 8 648 7
D 74 24 647 8 646 7
D 92 24 720 8 719 7
D 7026 24 11372 1200 11371 7
D 7095 24 11444 1200 11442 7
D 7164 24 11515 1200 11513 7
D 7233 24 11586 1200 11584 7
D 7302 24 11657 1288 11655 7
D 7377 24 11734 1288 11732 7
D 15492 21 12 1 3 18 0 0 0 0 0
 0 18 3 3 18 18
D 15495 21 12 1 3 18 0 0 0 0 0
 0 18 3 3 18 18
D 15498 21 12 1 3 18 0 0 0 0 0
 0 18 3 3 18 18
D 15501 21 12 1 3 18 0 0 0 0 0
 0 18 3 3 18 18
D 15504 24 31244 1984 31243 7
D 15510 21 9 2 11757 11756 0 1 0 0 1
 11746 11749 11754 11746 11749 11747
 11750 11753 11755 11750 11753 11751
D 15513 21 6 1 0 96 0 0 0 0 0
 0 96 0 3 96 0
D 15516 21 9 1 11766 11765 0 1 0 0 1
 11760 11763 11764 11760 11763 11761
D 15519 21 6 1 0 89 0 0 0 0 0
 0 89 0 3 89 0
D 15522 21 9 1 11775 11774 0 1 0 0 1
 11769 11772 11773 11769 11772 11770
D 15525 21 6 1 0 89 0 0 0 0 0
 0 89 0 3 89 0
D 15528 21 9 2 11790 11789 0 1 0 0 1
 11779 11782 11787 11779 11782 11780
 11783 11786 11788 11783 11786 11784
D 15531 21 6 1 0 96 0 0 0 0 0
 0 96 0 3 96 0
D 15534 21 9 1 11799 11798 0 1 0 0 1
 11793 11796 11797 11793 11796 11794
D 15537 21 6 1 0 89 0 0 0 0 0
 0 89 0 3 89 0
D 15540 21 9 1 11808 11807 0 1 0 0 1
 11802 11805 11806 11802 11805 11803
D 15543 21 6 1 0 89 0 0 0 0 0
 0 89 0 3 89 0
D 15546 21 9 1 11817 11816 0 1 0 0 1
 11811 11814 11815 11811 11814 11812
D 15549 21 6 1 0 89 0 0 0 0 0
 0 89 0 3 89 0
D 15552 21 9 2 11832 11831 0 1 0 0 1
 11821 11824 11829 11821 11824 11822
 11825 11828 11830 11825 11828 11826
D 15555 21 6 1 0 96 0 0 0 0 0
 0 96 0 3 96 0
D 15566 21 6 1 0 3 0 0 0 0 0
 0 3 0 3 3 0
D 15569 21 9 1 11833 11836 1 1 0 0 1
 3 11834 3 3 11834 11835
D 15572 21 6 1 11837 11840 1 1 0 0 1
 3 11838 3 3 11838 11839
D 15575 21 6 1 11841 11844 1 1 0 0 1
 3 11842 3 3 11842 11843
D 15578 21 9 1 11845 11848 1 1 0 0 1
 3 11846 3 3 11846 11847
D 15581 21 9 1 3 11850 0 0 1 0 0
 0 11849 3 3 11850 11850
D 15584 21 9 1 11851 11854 1 1 0 0 1
 3 11852 3 3 11852 11853
D 15587 21 9 1 11855 11858 1 1 0 0 1
 3 11856 3 3 11856 11857
D 15590 21 9 1 3 11860 0 0 1 0 0
 0 11859 3 3 11860 11860
D 15593 21 9 1 11861 11864 1 1 0 0 1
 3 11862 3 3 11862 11863
D 15596 21 9 1 11865 11868 1 1 0 0 1
 3 11866 3 3 11866 11867
D 15599 21 9 1 11869 11872 1 1 0 0 1
 3 11870 3 3 11870 11871
D 15602 21 9 1 11873 11876 1 1 0 0 1
 3 11874 3 3 11874 11875
D 15605 21 9 1 11877 11880 1 1 0 0 1
 3 11878 3 3 11878 11879
D 15608 21 9 1 11881 11884 1 1 0 0 1
 3 11882 3 3 11882 11883
D 15611 21 9 1 11885 11888 1 1 0 0 1
 3 11886 3 3 11886 11887
D 15614 21 9 1 11889 11892 1 1 0 0 1
 3 11890 3 3 11890 11891
D 15617 21 9 1 11893 11896 1 1 0 0 1
 3 11894 3 3 11894 11895
D 15620 21 9 1 11897 11900 1 1 0 0 1
 3 11898 3 3 11898 11899
D 15623 21 9 1 11901 11904 1 1 0 0 1
 3 11902 3 3 11902 11903
S 624 24 0 0 0 6 1 0 5012 10005 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 matexponential
S 632 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
R 646 25 6 iso_c_binding c_ptr
R 647 5 7 iso_c_binding val c_ptr
R 648 25 8 iso_c_binding c_funptr
R 649 5 9 iso_c_binding val c_funptr
R 682 6 42 iso_c_binding c_null_ptr$ac
R 684 6 44 iso_c_binding c_null_funptr$ac
R 685 26 45 iso_c_binding ==
R 687 26 47 iso_c_binding !=
R 719 25 5 pgi_acc_common c_devptr
R 720 5 6 pgi_acc_common cptr c_devptr
R 722 6 8 pgi_acc_common c_null_devptr$ac
R 736 14 22 pgi_acc_common __pgf90_assign_int_to_dim3
S 778 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 779 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 783 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 785 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 788 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 13 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 789 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 24 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 790 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 19 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 791 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 20 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 792 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 23 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 801 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 11370 3 0 0 0 16 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16
R 11371 25 1 csrmatrix csr_float
R 11372 5 2 csrmatrix csrval csr_float
R 11374 5 4 csrmatrix csrval$sd csr_float
R 11375 5 5 csrmatrix csrval$p csr_float
R 11376 5 6 csrmatrix csrval$o csr_float
R 11378 5 8 csrmatrix csrrowptr csr_float
R 11380 5 10 csrmatrix csrrowptr$sd csr_float
R 11381 5 11 csrmatrix csrrowptr$p csr_float
R 11382 5 12 csrmatrix csrrowptr$o csr_float
R 11384 5 14 csrmatrix csrcolidx csr_float
R 11386 5 16 csrmatrix csrcolidx$sd csr_float
R 11387 5 17 csrmatrix csrcolidx$p csr_float
R 11388 5 18 csrmatrix csrcolidx$o csr_float
R 11390 5 20 csrmatrix bufval csr_float
R 11393 5 23 csrmatrix bufval$sd csr_float
R 11394 5 24 csrmatrix bufval$p csr_float
R 11395 5 25 csrmatrix bufval$o csr_float
R 11397 5 27 csrmatrix bufrowptr csr_float
R 11400 5 30 csrmatrix bufrowptr$sd csr_float
R 11401 5 31 csrmatrix bufrowptr$p csr_float
R 11402 5 32 csrmatrix bufrowptr$o csr_float
R 11404 5 34 csrmatrix bufcolidx csr_float
R 11407 5 37 csrmatrix bufcolidx$sd csr_float
R 11408 5 38 csrmatrix bufcolidx$p csr_float
R 11409 5 39 csrmatrix bufcolidx$o csr_float
R 11411 5 41 csrmatrix d_csrval csr_float
R 11413 5 43 csrmatrix d_csrval$sd csr_float
R 11414 5 44 csrmatrix d_csrval$p csr_float
R 11415 5 45 csrmatrix d_csrval$o csr_float
R 11417 5 47 csrmatrix d_csrrowptr csr_float
R 11419 5 49 csrmatrix d_csrrowptr$sd csr_float
R 11420 5 50 csrmatrix d_csrrowptr$p csr_float
R 11421 5 51 csrmatrix d_csrrowptr$o csr_float
R 11423 5 53 csrmatrix d_csrcolidx csr_float
R 11425 5 55 csrmatrix d_csrcolidx$sd csr_float
R 11426 5 56 csrmatrix d_csrcolidx$p csr_float
R 11427 5 57 csrmatrix d_csrcolidx$o csr_float
R 11429 5 59 csrmatrix info csr_float
R 11430 5 60 csrmatrix infolu csr_float
R 11431 5 61 csrmatrix descr csr_float
R 11432 5 62 csrmatrix descrlu csr_float
R 11433 5 63 csrmatrix nr csr_float
R 11434 5 64 csrmatrix nc csr_float
R 11435 5 65 csrmatrix nnz csr_float
R 11436 5 66 csrmatrix nnzomp csr_float
R 11437 5 67 csrmatrix nthread csr_float
R 11438 5 68 csrmatrix lalloc csr_float
R 11439 5 69 csrmatrix lfinalized csr_float
R 11440 5 70 csrmatrix ldevice csr_float
R 11441 5 71 csrmatrix lparallel csr_float
R 11442 25 72 csrmatrix csr_float_complex
R 11444 5 74 csrmatrix csrval csr_float_complex
R 11445 5 75 csrmatrix csrval$sd csr_float_complex
R 11446 5 76 csrmatrix csrval$p csr_float_complex
R 11447 5 77 csrmatrix csrval$o csr_float_complex
R 11450 5 80 csrmatrix csrrowptr csr_float_complex
R 11451 5 81 csrmatrix csrrowptr$sd csr_float_complex
R 11452 5 82 csrmatrix csrrowptr$p csr_float_complex
R 11453 5 83 csrmatrix csrrowptr$o csr_float_complex
R 11456 5 86 csrmatrix csrcolidx csr_float_complex
R 11457 5 87 csrmatrix csrcolidx$sd csr_float_complex
R 11458 5 88 csrmatrix csrcolidx$p csr_float_complex
R 11459 5 89 csrmatrix csrcolidx$o csr_float_complex
R 11463 5 93 csrmatrix bufval csr_float_complex
R 11464 5 94 csrmatrix bufval$sd csr_float_complex
R 11465 5 95 csrmatrix bufval$p csr_float_complex
R 11466 5 96 csrmatrix bufval$o csr_float_complex
R 11470 5 100 csrmatrix bufrowptr csr_float_complex
R 11471 5 101 csrmatrix bufrowptr$sd csr_float_complex
R 11472 5 102 csrmatrix bufrowptr$p csr_float_complex
R 11473 5 103 csrmatrix bufrowptr$o csr_float_complex
R 11477 5 107 csrmatrix bufcolidx csr_float_complex
R 11478 5 108 csrmatrix bufcolidx$sd csr_float_complex
R 11479 5 109 csrmatrix bufcolidx$p csr_float_complex
R 11480 5 110 csrmatrix bufcolidx$o csr_float_complex
R 11483 5 113 csrmatrix d_csrval csr_float_complex
R 11484 5 114 csrmatrix d_csrval$sd csr_float_complex
R 11485 5 115 csrmatrix d_csrval$p csr_float_complex
R 11486 5 116 csrmatrix d_csrval$o csr_float_complex
R 11489 5 119 csrmatrix d_csrrowptr csr_float_complex
R 11490 5 120 csrmatrix d_csrrowptr$sd csr_float_complex
R 11491 5 121 csrmatrix d_csrrowptr$p csr_float_complex
R 11492 5 122 csrmatrix d_csrrowptr$o csr_float_complex
R 11495 5 125 csrmatrix d_csrcolidx csr_float_complex
R 11496 5 126 csrmatrix d_csrcolidx$sd csr_float_complex
R 11497 5 127 csrmatrix d_csrcolidx$p csr_float_complex
R 11498 5 128 csrmatrix d_csrcolidx$o csr_float_complex
R 11500 5 130 csrmatrix info csr_float_complex
R 11501 5 131 csrmatrix infolu csr_float_complex
R 11502 5 132 csrmatrix descr csr_float_complex
R 11503 5 133 csrmatrix descrlu csr_float_complex
R 11504 5 134 csrmatrix nr csr_float_complex
R 11505 5 135 csrmatrix nc csr_float_complex
R 11506 5 136 csrmatrix nnz csr_float_complex
R 11507 5 137 csrmatrix nnzomp csr_float_complex
R 11508 5 138 csrmatrix nthread csr_float_complex
R 11509 5 139 csrmatrix lalloc csr_float_complex
R 11510 5 140 csrmatrix lfinalized csr_float_complex
R 11511 5 141 csrmatrix ldevice csr_float_complex
R 11512 5 142 csrmatrix lparallel csr_float_complex
R 11513 25 143 csrmatrix csr_double
R 11515 5 145 csrmatrix csrval csr_double
R 11516 5 146 csrmatrix csrval$sd csr_double
R 11517 5 147 csrmatrix csrval$p csr_double
R 11518 5 148 csrmatrix csrval$o csr_double
R 11521 5 151 csrmatrix csrrowptr csr_double
R 11522 5 152 csrmatrix csrrowptr$sd csr_double
R 11523 5 153 csrmatrix csrrowptr$p csr_double
R 11524 5 154 csrmatrix csrrowptr$o csr_double
R 11527 5 157 csrmatrix csrcolidx csr_double
R 11528 5 158 csrmatrix csrcolidx$sd csr_double
R 11529 5 159 csrmatrix csrcolidx$p csr_double
R 11530 5 160 csrmatrix csrcolidx$o csr_double
R 11534 5 164 csrmatrix bufval csr_double
R 11535 5 165 csrmatrix bufval$sd csr_double
R 11536 5 166 csrmatrix bufval$p csr_double
R 11537 5 167 csrmatrix bufval$o csr_double
R 11541 5 171 csrmatrix bufrowptr csr_double
R 11542 5 172 csrmatrix bufrowptr$sd csr_double
R 11543 5 173 csrmatrix bufrowptr$p csr_double
R 11544 5 174 csrmatrix bufrowptr$o csr_double
R 11548 5 178 csrmatrix bufcolidx csr_double
R 11549 5 179 csrmatrix bufcolidx$sd csr_double
R 11550 5 180 csrmatrix bufcolidx$p csr_double
R 11551 5 181 csrmatrix bufcolidx$o csr_double
R 11554 5 184 csrmatrix d_csrval csr_double
R 11555 5 185 csrmatrix d_csrval$sd csr_double
R 11556 5 186 csrmatrix d_csrval$p csr_double
R 11557 5 187 csrmatrix d_csrval$o csr_double
R 11560 5 190 csrmatrix d_csrrowptr csr_double
R 11561 5 191 csrmatrix d_csrrowptr$sd csr_double
R 11562 5 192 csrmatrix d_csrrowptr$p csr_double
R 11563 5 193 csrmatrix d_csrrowptr$o csr_double
R 11566 5 196 csrmatrix d_csrcolidx csr_double
R 11567 5 197 csrmatrix d_csrcolidx$sd csr_double
R 11568 5 198 csrmatrix d_csrcolidx$p csr_double
R 11569 5 199 csrmatrix d_csrcolidx$o csr_double
R 11571 5 201 csrmatrix info csr_double
R 11572 5 202 csrmatrix infolu csr_double
R 11573 5 203 csrmatrix descr csr_double
R 11574 5 204 csrmatrix descrlu csr_double
R 11575 5 205 csrmatrix nr csr_double
R 11576 5 206 csrmatrix nc csr_double
R 11577 5 207 csrmatrix nnz csr_double
R 11578 5 208 csrmatrix nnzomp csr_double
R 11579 5 209 csrmatrix nthread csr_double
R 11580 5 210 csrmatrix lalloc csr_double
R 11581 5 211 csrmatrix lfinalized csr_double
R 11582 5 212 csrmatrix ldevice csr_double
R 11583 5 213 csrmatrix lparallel csr_double
R 11584 25 214 csrmatrix csr_double_complex
R 11586 5 216 csrmatrix csrval csr_double_complex
R 11587 5 217 csrmatrix csrval$sd csr_double_complex
R 11588 5 218 csrmatrix csrval$p csr_double_complex
R 11589 5 219 csrmatrix csrval$o csr_double_complex
R 11592 5 222 csrmatrix csrrowptr csr_double_complex
R 11593 5 223 csrmatrix csrrowptr$sd csr_double_complex
R 11594 5 224 csrmatrix csrrowptr$p csr_double_complex
R 11595 5 225 csrmatrix csrrowptr$o csr_double_complex
R 11598 5 228 csrmatrix csrcolidx csr_double_complex
R 11599 5 229 csrmatrix csrcolidx$sd csr_double_complex
R 11600 5 230 csrmatrix csrcolidx$p csr_double_complex
R 11601 5 231 csrmatrix csrcolidx$o csr_double_complex
R 11605 5 235 csrmatrix bufval csr_double_complex
R 11606 5 236 csrmatrix bufval$sd csr_double_complex
R 11607 5 237 csrmatrix bufval$p csr_double_complex
R 11608 5 238 csrmatrix bufval$o csr_double_complex
R 11612 5 242 csrmatrix bufrowptr csr_double_complex
R 11613 5 243 csrmatrix bufrowptr$sd csr_double_complex
R 11614 5 244 csrmatrix bufrowptr$p csr_double_complex
R 11615 5 245 csrmatrix bufrowptr$o csr_double_complex
R 11619 5 249 csrmatrix bufcolidx csr_double_complex
R 11620 5 250 csrmatrix bufcolidx$sd csr_double_complex
R 11621 5 251 csrmatrix bufcolidx$p csr_double_complex
R 11622 5 252 csrmatrix bufcolidx$o csr_double_complex
R 11625 5 255 csrmatrix d_csrval csr_double_complex
R 11626 5 256 csrmatrix d_csrval$sd csr_double_complex
R 11627 5 257 csrmatrix d_csrval$p csr_double_complex
R 11628 5 258 csrmatrix d_csrval$o csr_double_complex
R 11631 5 261 csrmatrix d_csrrowptr csr_double_complex
R 11632 5 262 csrmatrix d_csrrowptr$sd csr_double_complex
R 11633 5 263 csrmatrix d_csrrowptr$p csr_double_complex
R 11634 5 264 csrmatrix d_csrrowptr$o csr_double_complex
R 11637 5 267 csrmatrix d_csrcolidx csr_double_complex
R 11638 5 268 csrmatrix d_csrcolidx$sd csr_double_complex
R 11639 5 269 csrmatrix d_csrcolidx$p csr_double_complex
R 11640 5 270 csrmatrix d_csrcolidx$o csr_double_complex
R 11642 5 272 csrmatrix info csr_double_complex
R 11643 5 273 csrmatrix infolu csr_double_complex
R 11644 5 274 csrmatrix descr csr_double_complex
R 11645 5 275 csrmatrix descrlu csr_double_complex
R 11646 5 276 csrmatrix nr csr_double_complex
R 11647 5 277 csrmatrix nc csr_double_complex
R 11648 5 278 csrmatrix nnz csr_double_complex
R 11649 5 279 csrmatrix nnzomp csr_double_complex
R 11650 5 280 csrmatrix nthread csr_double_complex
R 11651 5 281 csrmatrix lalloc csr_double_complex
R 11652 5 282 csrmatrix lfinalized csr_double_complex
R 11653 5 283 csrmatrix ldevice csr_double_complex
R 11654 5 284 csrmatrix lparallel csr_double_complex
R 11655 25 285 csrmatrix csr_mixed
R 11657 5 287 csrmatrix csrval csr_mixed
R 11658 5 288 csrmatrix csrval$sd csr_mixed
R 11659 5 289 csrmatrix csrval$p csr_mixed
R 11660 5 290 csrmatrix csrval$o csr_mixed
R 11663 5 293 csrmatrix csrrowptr csr_mixed
R 11664 5 294 csrmatrix csrrowptr$sd csr_mixed
R 11665 5 295 csrmatrix csrrowptr$p csr_mixed
R 11666 5 296 csrmatrix csrrowptr$o csr_mixed
R 11669 5 299 csrmatrix csrcolidx csr_mixed
R 11670 5 300 csrmatrix csrcolidx$sd csr_mixed
R 11671 5 301 csrmatrix csrcolidx$p csr_mixed
R 11672 5 302 csrmatrix csrcolidx$o csr_mixed
R 11676 5 306 csrmatrix bufval csr_mixed
R 11677 5 307 csrmatrix bufval$sd csr_mixed
R 11678 5 308 csrmatrix bufval$p csr_mixed
R 11679 5 309 csrmatrix bufval$o csr_mixed
R 11683 5 313 csrmatrix bufrowptr csr_mixed
R 11684 5 314 csrmatrix bufrowptr$sd csr_mixed
R 11685 5 315 csrmatrix bufrowptr$p csr_mixed
R 11686 5 316 csrmatrix bufrowptr$o csr_mixed
R 11690 5 320 csrmatrix bufcolidx csr_mixed
R 11691 5 321 csrmatrix bufcolidx$sd csr_mixed
R 11692 5 322 csrmatrix bufcolidx$p csr_mixed
R 11693 5 323 csrmatrix bufcolidx$o csr_mixed
R 11696 5 326 csrmatrix d_csrval csr_mixed
R 11697 5 327 csrmatrix d_csrval$sd csr_mixed
R 11698 5 328 csrmatrix d_csrval$p csr_mixed
R 11699 5 329 csrmatrix d_csrval$o csr_mixed
R 11701 5 331 csrmatrix d_csrval8 csr_mixed
R 11703 5 333 csrmatrix d_csrval8$sd csr_mixed
R 11704 5 334 csrmatrix d_csrval8$p csr_mixed
R 11705 5 335 csrmatrix d_csrval8$o csr_mixed
R 11708 5 338 csrmatrix d_csrrowptr csr_mixed
R 11709 5 339 csrmatrix d_csrrowptr$sd csr_mixed
R 11710 5 340 csrmatrix d_csrrowptr$p csr_mixed
R 11711 5 341 csrmatrix d_csrrowptr$o csr_mixed
R 11714 5 344 csrmatrix d_csrcolidx csr_mixed
R 11715 5 345 csrmatrix d_csrcolidx$sd csr_mixed
R 11716 5 346 csrmatrix d_csrcolidx$p csr_mixed
R 11717 5 347 csrmatrix d_csrcolidx$o csr_mixed
R 11719 5 349 csrmatrix info csr_mixed
R 11720 5 350 csrmatrix infolu csr_mixed
R 11721 5 351 csrmatrix descr csr_mixed
R 11722 5 352 csrmatrix descrlu csr_mixed
R 11723 5 353 csrmatrix nr csr_mixed
R 11724 5 354 csrmatrix nc csr_mixed
R 11725 5 355 csrmatrix nnz csr_mixed
R 11726 5 356 csrmatrix nnzomp csr_mixed
R 11727 5 357 csrmatrix nthread csr_mixed
R 11728 5 358 csrmatrix lalloc csr_mixed
R 11729 5 359 csrmatrix lfinalized csr_mixed
R 11730 5 360 csrmatrix ldevice csr_mixed
R 11731 5 361 csrmatrix lparallel csr_mixed
R 11732 25 362 csrmatrix csr_mixed_complex
R 11734 5 364 csrmatrix csrval csr_mixed_complex
R 11735 5 365 csrmatrix csrval$sd csr_mixed_complex
R 11736 5 366 csrmatrix csrval$p csr_mixed_complex
R 11737 5 367 csrmatrix csrval$o csr_mixed_complex
R 11740 5 370 csrmatrix csrrowptr csr_mixed_complex
R 11741 5 371 csrmatrix csrrowptr$sd csr_mixed_complex
R 11742 5 372 csrmatrix csrrowptr$p csr_mixed_complex
R 11743 5 373 csrmatrix csrrowptr$o csr_mixed_complex
R 11746 5 376 csrmatrix csrcolidx csr_mixed_complex
R 11747 5 377 csrmatrix csrcolidx$sd csr_mixed_complex
R 11748 5 378 csrmatrix csrcolidx$p csr_mixed_complex
R 11749 5 379 csrmatrix csrcolidx$o csr_mixed_complex
R 11753 5 383 csrmatrix bufval csr_mixed_complex
R 11754 5 384 csrmatrix bufval$sd csr_mixed_complex
R 11755 5 385 csrmatrix bufval$p csr_mixed_complex
R 11756 5 386 csrmatrix bufval$o csr_mixed_complex
R 11760 5 390 csrmatrix bufrowptr csr_mixed_complex
R 11761 5 391 csrmatrix bufrowptr$sd csr_mixed_complex
R 11762 5 392 csrmatrix bufrowptr$p csr_mixed_complex
R 11763 5 393 csrmatrix bufrowptr$o csr_mixed_complex
R 11767 5 397 csrmatrix bufcolidx csr_mixed_complex
R 11768 5 398 csrmatrix bufcolidx$sd csr_mixed_complex
R 11769 5 399 csrmatrix bufcolidx$p csr_mixed_complex
R 11770 5 400 csrmatrix bufcolidx$o csr_mixed_complex
R 11773 5 403 csrmatrix d_csrval csr_mixed_complex
R 11774 5 404 csrmatrix d_csrval$sd csr_mixed_complex
R 11775 5 405 csrmatrix d_csrval$p csr_mixed_complex
R 11776 5 406 csrmatrix d_csrval$o csr_mixed_complex
R 11779 5 409 csrmatrix d_csrval8 csr_mixed_complex
R 11780 5 410 csrmatrix d_csrval8$sd csr_mixed_complex
R 11781 5 411 csrmatrix d_csrval8$p csr_mixed_complex
R 11782 5 412 csrmatrix d_csrval8$o csr_mixed_complex
R 11785 5 415 csrmatrix d_csrrowptr csr_mixed_complex
R 11786 5 416 csrmatrix d_csrrowptr$sd csr_mixed_complex
R 11787 5 417 csrmatrix d_csrrowptr$p csr_mixed_complex
R 11788 5 418 csrmatrix d_csrrowptr$o csr_mixed_complex
R 11791 5 421 csrmatrix d_csrcolidx csr_mixed_complex
R 11792 5 422 csrmatrix d_csrcolidx$sd csr_mixed_complex
R 11793 5 423 csrmatrix d_csrcolidx$p csr_mixed_complex
R 11794 5 424 csrmatrix d_csrcolidx$o csr_mixed_complex
R 11796 5 426 csrmatrix info csr_mixed_complex
R 11797 5 427 csrmatrix infolu csr_mixed_complex
R 11798 5 428 csrmatrix descr csr_mixed_complex
R 11799 5 429 csrmatrix descrlu csr_mixed_complex
R 11800 5 430 csrmatrix nr csr_mixed_complex
R 11801 5 431 csrmatrix nc csr_mixed_complex
R 11802 5 432 csrmatrix nnz csr_mixed_complex
R 11803 5 433 csrmatrix nnzomp csr_mixed_complex
R 11804 5 434 csrmatrix nthread csr_mixed_complex
R 11805 5 435 csrmatrix lalloc csr_mixed_complex
R 11806 5 436 csrmatrix lfinalized csr_mixed_complex
R 11807 5 437 csrmatrix ldevice csr_mixed_complex
R 11808 5 438 csrmatrix lparallel csr_mixed_complex
R 11894 26 524 csrmatrix -
R 12213 14 843 csrmatrix copycsrfloat
R 12217 14 847 csrmatrix copycsrdouble
R 12221 14 851 csrmatrix copycsrmixed
R 12225 14 855 csrmatrix copycsrfloatz
R 12229 14 859 csrmatrix copycsrdoublez
R 12233 14 863 csrmatrix copycsrmixedz
R 12237 14 867 csrmatrix copycsrfloat2double
R 12241 14 871 csrmatrix copycsrfloat2mixed
R 12245 14 875 csrmatrix copycsrdouble2float
R 12249 14 879 csrmatrix copycsrdouble2mixed
R 12253 14 883 csrmatrix copycsrmixed2float
R 12257 14 887 csrmatrix copycsrmixed2double
R 12261 14 891 csrmatrix copycsrfloat2floatz
R 12265 14 895 csrmatrix copycsrfloat2doublez
R 12269 14 899 csrmatrix copycsrfloat2mixedz
R 12273 14 903 csrmatrix copycsrdouble2floatz
R 12277 14 907 csrmatrix copycsrdouble2doublez
R 12281 14 911 csrmatrix copycsrdouble2mixedz
R 12285 14 915 csrmatrix copycsrmixed2floatz
R 12289 14 919 csrmatrix copycsrmixed2doublez
R 12293 14 923 csrmatrix copycsrmixed2mixedz
R 12297 14 927 csrmatrix copycsrfloatz2doublez
R 12301 14 931 csrmatrix copycsrfloatz2mixedz
R 12305 14 935 csrmatrix copycsrdoublez2floatz
R 12309 14 939 csrmatrix copycsrdoublez2mixedz
R 12313 14 943 csrmatrix copycsrmixedz2floatz
R 12317 14 947 csrmatrix copycsrmixedz2doublez
R 31183 14 8728 cublas assign_ptr2_cublashandles
S 31188 16 0 0 0 6 1 624 123167 4 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 16 101 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 cram_order
S 31189 16 0 0 0 15492 1 624 123178 4 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31214 10 0 0 0 0 0 0 0 0 0 0 0 11721 0 624 0 0 0 0 pole
S 31190 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1071271915 -432335825 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31191 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1077102342 -1236897252 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31192 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31190 31191 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31193 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1072361644 -1512449415 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31194 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1076901984 1866015257 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31195 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31193 31194 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31196 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1075301096 1839929361 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31197 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1074574108 -663925227 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31198 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31196 31197 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31199 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1074533028 -375171232 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31200 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1075896149 2033571824 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31201 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31199 31200 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31202 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1075423786 2090587023 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31203 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1072896800 355410633 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31204 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31202 31203 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31205 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1073132995 -2144187875 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31206 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1076222409 1042130669 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31207 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31205 31206 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31208 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1075050754 -865674940 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31209 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1075313870 -1890081998 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31210 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31208 31209 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31211 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1074356365 1939103630 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31212 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1076559573 -417448187 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31213 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31211 31212 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31214 7 4 0 0 15492 31240 624 123548 4080004c 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 31300 0 0 0 0 0 0 0 0 11721 0 624 0 31189 0 0 pole$ac
S 31215 16 0 0 0 15498 1 624 16822 4 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31240 11 0 0 0 0 0 0 0 0 0 0 0 11739 0 624 0 0 0 0 res
S 31216 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1096739736 1302432435 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31217 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1090951806 -259241253 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31218 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31216 31217 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31219 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1059830104 -779354864 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31220 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1064434306 -290988849 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31221 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31219 31220 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31222 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1079794036 -1008274466 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31223 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1079606431 876253756 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31224 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31222 31223 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31225 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1076764289 -169597453 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31226 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1072234128 1654001493 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31227 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31225 31226 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31228 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1068490738 1656012437 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31229 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1066659067 1663339729 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31230 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31228 31229 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31231 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1074287817 956534826 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31232 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1073499245 875399812 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31233 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31231 31232 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31234 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1068547494 -1356202938 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31235 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1071226502 -251213104 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31236 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31234 31235 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31237 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1067778300 237814318 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31238 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 -1077664047 -1447817244 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31239 3 0 0 0 12 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31237 31238 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12
S 31240 7 4 0 0 15498 1 624 123949 4080004c 400000 A 0 0 0 0 B 0 0 0 0 0 128 0 0 0 0 0 0 31300 0 0 0 0 0 0 0 0 11739 0 624 0 31215 0 0 res$ac
S 31241 16 0 0 0 9 1 624 123956 4 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 31242 11741 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 res0
S 31242 3 0 0 0 9 0 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 1018077011 -1766568286 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 9
S 31243 25 0 0 0 15504 1 624 123987 1000000c 800054 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 31299 0 0 0 624 0 0 0 0 matexpvars_type
S 31244 5 0 0 0 16 31245 624 124003 800004 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 1 31244 0 624 0 0 0 0 loutmat
S 31245 5 0 0 0 6 31246 624 124011 800004 0 A 0 0 0 0 B 0 0 0 0 0 4 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31244 31245 0 624 0 0 0 0 rank
S 31246 5 6 0 0 15510 31250 624 5757 10a00004 14 A 0 0 0 0 B 0 0 0 0 0 8 31250 0 15504 0 31252 0 0 0 0 0 0 0 0 31249 31245 31246 31251 624 0 0 0 0 a
S 31247 6 4 0 0 6 31248 624 46705 40800006 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_0_1
S 31248 6 4 0 0 6 31254 624 46713 40800006 0 A 0 0 0 0 B 0 0 0 0 0 4 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_1_1
S 31249 5 0 0 0 15513 31253 624 124016 40822004 1020 A 0 0 0 0 B 0 0 0 0 0 24 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31251 31249 0 624 0 0 0 0 a$sd
S 31250 5 0 0 0 7 31251 624 124021 40802001 1020 A 0 0 0 0 B 0 0 0 0 0 8 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31246 31250 0 624 0 0 0 0 a$p
S 31251 5 0 0 0 7 31249 624 124025 40802000 1020 A 0 0 0 0 B 0 0 0 0 0 16 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31250 31251 0 624 0 0 0 0 a$o
S 31252 22 1 0 0 9 1 624 124029 40000000 1000 A 0 0 0 0 B 0 0 0 0 0 0 0 31246 0 0 0 0 31249 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 a$arrdsc
S 31253 5 6 0 0 15516 31256 624 124038 10a00004 14 A 0 0 0 0 B 0 0 0 0 0 120 31256 0 15504 0 31258 0 0 0 0 0 0 0 0 31255 31246 31253 31257 624 0 0 0 0 x0
S 31254 6 4 0 0 6 31260 624 46721 40800006 0 A 0 0 0 0 B 0 0 0 0 0 8 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_2_1
S 31255 5 0 0 0 15519 31259 624 124041 40822004 1020 A 0 0 0 0 B 0 0 0 0 0 136 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31257 31255 0 624 0 0 0 0 x0$sd
S 31256 5 0 0 0 7 31257 624 124047 40802001 1020 A 0 0 0 0 B 0 0 0 0 0 120 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31253 31256 0 624 0 0 0 0 x0$p
S 31257 5 0 0 0 7 31255 624 124052 40802000 1020 A 0 0 0 0 B 0 0 0 0 0 128 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31256 31257 0 624 0 0 0 0 x0$o
S 31258 22 1 0 0 9 1 624 124057 40000000 1000 A 0 0 0 0 B 0 0 0 0 0 0 0 31253 0 0 0 0 31255 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 x0$arrdsc
S 31259 5 6 0 0 15522 31262 624 110784 10a00004 14 A 0 0 0 0 B 0 0 0 0 0 208 31262 0 15504 0 31264 0 0 0 0 0 0 0 0 31261 31253 31259 31263 624 0 0 0 0 x1
S 31260 6 4 0 0 6 31266 624 46729 40800006 0 A 0 0 0 0 B 0 0 0 0 0 12 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_3_1
S 31261 5 0 0 0 15525 31265 624 124067 40822004 1020 A 0 0 0 0 B 0 0 0 0 0 224 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31263 31261 0 624 0 0 0 0 x1$sd
S 31262 5 0 0 0 7 31263 624 124073 40802001 1020 A 0 0 0 0 B 0 0 0 0 0 208 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31259 31262 0 624 0 0 0 0 x1$p
S 31263 5 0 0 0 7 31261 624 124078 40802000 1020 A 0 0 0 0 B 0 0 0 0 0 216 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31262 31263 0 624 0 0 0 0 x1$o
S 31264 22 1 0 0 9 1 624 124083 40000000 1000 A 0 0 0 0 B 0 0 0 0 0 0 0 31259 0 0 0 0 31261 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 x1$arrdsc
S 31265 5 6 0 0 15528 31269 624 124093 10a00004 14 A 0 0 0 0 B 0 0 0 0 0 296 31269 0 15504 0 31271 0 0 0 0 0 0 0 0 31268 31259 31265 31270 624 0 0 0 0 expa
S 31266 6 4 0 0 6 31267 624 46737 40800006 0 A 0 0 0 0 B 0 0 0 0 0 16 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_4_1
S 31267 6 4 0 0 6 31274 624 46745 40800006 0 A 0 0 0 0 B 0 0 0 0 0 20 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_5_1
S 31268 5 0 0 0 15531 31272 624 124098 40822004 1020 A 0 0 0 0 B 0 0 0 0 0 312 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31270 31268 0 624 0 0 0 0 expa$sd
S 31269 5 0 0 0 7 31270 624 124106 40802001 1020 A 0 0 0 0 B 0 0 0 0 0 296 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31265 31269 0 624 0 0 0 0 expa$p
S 31270 5 0 0 0 7 31268 624 124113 40802000 1020 A 0 0 0 0 B 0 0 0 0 0 304 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31269 31270 0 624 0 0 0 0 expa$o
S 31271 22 1 0 0 9 1 624 124120 40000000 1000 A 0 0 0 0 B 0 0 0 0 0 0 0 31265 0 0 0 0 31268 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 expa$arrdsc
S 31272 5 0 0 0 7164 31273 624 124132 800004 0 A 0 0 0 0 B 0 0 0 0 0 408 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31265 31272 0 624 0 0 0 0 a_csr
S 31273 5 6 0 0 15534 31276 624 112571 10a08004 51 A 0 0 0 0 B 0 0 0 0 0 1608 31276 0 15504 0 31278 0 0 0 0 0 0 0 0 31275 31272 31273 31277 624 0 0 0 0 da
S 31274 6 4 0 0 6 31280 624 46753 40800006 0 A 0 0 0 0 B 0 0 0 0 0 24 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_6_1
S 31275 5 0 0 0 15537 31279 624 124138 40822004 1020 A 0 0 0 0 B 0 0 0 0 0 1624 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31277 31275 0 624 0 0 0 0 da$sd
S 31276 5 0 0 0 7 31277 624 124144 40802001 1020 A 0 0 0 0 B 0 0 0 0 0 1608 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31273 31276 0 624 0 0 0 0 da$p
S 31277 5 0 0 0 7 31275 624 124149 40802000 1020 A 0 0 0 0 B 0 0 0 0 0 1616 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31276 31277 0 624 0 0 0 0 da$o
S 31278 22 1 0 0 9 1 624 124154 40000000 1000 A 0 0 0 0 B 0 0 0 0 0 0 0 31273 0 0 0 0 31275 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 da$arrdsc
S 31279 5 6 0 0 15540 31282 624 124164 10a08004 51 A 0 0 0 0 B 0 0 0 0 0 1696 31282 0 15504 0 31284 0 0 0 0 0 0 0 0 31281 31273 31279 31283 624 0 0 0 0 dx0
S 31280 6 4 0 0 6 31286 624 46761 40800006 0 A 0 0 0 0 B 0 0 0 0 0 28 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_7_1
S 31281 5 0 0 0 15543 31285 624 124168 40822004 1020 A 0 0 0 0 B 0 0 0 0 0 1712 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31283 31281 0 624 0 0 0 0 dx0$sd
S 31282 5 0 0 0 7 31283 624 124175 40802001 1020 A 0 0 0 0 B 0 0 0 0 0 1696 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31279 31282 0 624 0 0 0 0 dx0$p
S 31283 5 0 0 0 7 31281 624 124181 40802000 1020 A 0 0 0 0 B 0 0 0 0 0 1704 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31282 31283 0 624 0 0 0 0 dx0$o
S 31284 22 1 0 0 9 1 624 124187 40000000 1000 A 0 0 0 0 B 0 0 0 0 0 0 0 31279 0 0 0 0 31281 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 dx0$arrdsc
S 31285 5 6 0 0 15546 31288 624 124198 10a08004 51 A 0 0 0 0 B 0 0 0 0 0 1784 31288 0 15504 0 31290 0 0 0 0 0 0 0 0 31287 31279 31285 31289 624 0 0 0 0 dx1
S 31286 6 4 0 0 6 31292 624 46769 40800006 0 A 0 0 0 0 B 0 0 0 0 0 32 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_8_1
S 31287 5 0 0 0 15549 31291 624 124202 40822004 1020 A 0 0 0 0 B 0 0 0 0 0 1800 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31289 31287 0 624 0 0 0 0 dx1$sd
S 31288 5 0 0 0 7 31289 624 124209 40802001 1020 A 0 0 0 0 B 0 0 0 0 0 1784 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31285 31288 0 624 0 0 0 0 dx1$p
S 31289 5 0 0 0 7 31287 624 124215 40802000 1020 A 0 0 0 0 B 0 0 0 0 0 1792 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31288 31289 0 624 0 0 0 0 dx1$o
S 31290 22 1 0 0 9 1 624 124221 40000000 1000 A 0 0 0 0 B 0 0 0 0 0 0 0 31285 0 0 0 0 31287 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 dx1$arrdsc
S 31291 5 6 0 0 15552 31295 624 124232 10a08004 51 A 0 0 0 0 B 0 0 0 0 0 1872 31295 0 15504 0 31297 0 0 0 0 0 0 0 0 31294 31285 31291 31296 624 0 0 0 0 dexpa
S 31292 6 4 0 0 6 31293 624 124238 40800006 0 A 0 0 0 0 B 0 0 0 0 0 36 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_9_1
S 31293 6 4 0 0 6 1 624 124246 40800006 0 A 0 0 0 0 B 0 0 0 0 0 40 0 0 0 0 0 0 31301 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 z_b_10_1
S 31294 5 0 0 0 15555 1 624 124255 40822004 1020 A 0 0 0 0 B 0 0 0 0 0 1888 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31296 31294 0 624 0 0 0 0 dexpa$sd
S 31295 5 0 0 0 7 31296 624 124264 40802001 1020 A 0 0 0 0 B 0 0 0 0 0 1872 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31291 31295 0 624 0 0 0 0 dexpa$p
S 31296 5 0 0 0 7 31294 624 124272 40802000 1020 A 0 0 0 0 B 0 0 0 0 0 1880 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 31295 31296 0 624 0 0 0 0 dexpa$o
S 31297 22 1 0 0 9 1 624 124280 40000000 1000 A 0 0 0 0 B 0 0 0 0 0 0 0 31291 0 0 0 0 31294 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 dexpa$arrdsc
S 31298 26 0 0 0 0 1 624 5870 4 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1918 29 0 0 0 0 0 624 0 0 0 0 =
O 31298 29 12317 12313 12309 12305 12301 12297 12293 12289 12285 12281 12277 12273 12269 12265 12261 12257 12253 12249 12245 12241 12237 12233 12229 12225 12221 12217 12213 736 31183
S 31299 8 5 0 0 15566 1 624 124293 40022004 1220 A 0 0 0 0 B 0 0 0 0 0 0 0 15504 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 624 0 0 0 0 matexponential$matexpvars_type$td
S 31300 11 0 0 0 9 11909 624 124327 40800000 805000 A 0 0 0 0 B 0 0 0 0 0 256 0 0 31214 31240 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _matexponential$8
S 31301 11 0 0 0 9 31300 624 124345 40800000 805000 A 0 0 0 0 B 0 0 0 0 0 44 0 0 31247 31293 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 _matexponential$0
S 31302 23 5 0 0 0 31304 624 124363 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 matexpkrylov_cuda
S 31303 1 3 0 0 15504 1 31302 124381 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 krylovvars
S 31304 14 5 0 0 0 1 31302 124363 0 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 18194 1 0 0 0 0 0 0 0 0 0 0 0 0 56 0 624 0 0 0 0 matexpkrylov_cuda
F 31304 1 31303
S 31305 23 5 0 4 0 31314 624 124392 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 cumv_csr
S 31306 7 3 0 0 15569 1 31305 5138 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 val
S 31307 7 3 0 0 15572 1 31305 124401 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 rowptr
S 31308 7 3 0 0 15575 1 31305 124408 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 colidx
S 31309 1 3 0 0 6 1 31305 16434 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 nnz
S 31310 6 3 0 0 6 1 31305 47346 808004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 nr
S 31311 1 3 0 0 6 1 31305 47349 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 nc
S 31312 7 3 0 0 15578 1 31305 124415 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v0
S 31313 7 3 3 0 15581 1 31305 124418 808204 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v1
S 31314 14 5 0 4 0 1 31305 124392 20000200 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 18196 8 0 0 0 0 0 0 0 0 0 0 0 0 304 0 624 0 0 0 0 cumv_csr
F 31314 8 31306 31307 31308 31309 31310 31311 31312 31313
S 31315 6 1 0 0 6 1 31305 124421 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_2
S 31316 6 1 0 0 6 1 31305 124429 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_2
S 31317 6 1 0 0 6 1 31305 124437 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_2
S 31318 6 1 0 0 6 1 31305 124445 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12099
S 31319 6 1 0 0 6 1 31305 124455 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_4_2
S 31320 6 1 0 0 6 1 31305 124463 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_2
S 31321 6 1 0 0 6 1 31305 124471 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_7_2
S 31322 6 1 0 0 6 1 31305 124479 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12106
S 31323 6 1 0 0 6 1 31305 124489 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_8_2
S 31324 6 1 0 0 6 1 31305 124497 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_10_2
S 31325 6 1 0 0 6 1 31305 124506 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_11_1
S 31326 6 1 0 0 6 1 31305 124515 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12113
S 31327 6 1 0 0 6 1 31305 124525 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_12_1
S 31328 6 1 0 0 6 1 31305 124534 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_14_1
S 31329 6 1 0 0 6 1 31305 124543 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_15_1
S 31330 6 1 0 0 6 1 31305 124552 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12120
S 31331 6 1 0 0 6 1 31305 124562 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12122
S 31332 23 5 0 4 0 31338 624 124572 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 cumv
S 31333 7 3 0 0 15584 1 31332 5757 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 31334 7 3 0 0 15587 1 31332 124415 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v0
S 31335 6 3 0 0 6 1 31332 19100 808004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 m
S 31336 1 3 0 0 6 1 31332 19102 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 n
S 31337 7 3 0 0 15590 1 31332 124418 808204 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v1
S 31338 14 5 0 4 0 1 31332 124572 20000200 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 18205 5 0 0 0 0 0 0 0 0 0 0 0 0 361 0 624 0 0 0 0 cumv
F 31338 5 31333 31334 31335 31336 31337
S 31339 6 1 0 0 6 1 31332 124421 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_2
S 31340 6 1 0 0 6 1 31332 124429 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_2
S 31341 6 1 0 0 6 1 31332 124437 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_2
S 31342 6 1 0 0 6 1 31332 124577 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12117
S 31343 6 1 0 0 6 1 31332 124455 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_4_2
S 31344 6 1 0 0 6 1 31332 124463 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_2
S 31345 6 1 0 0 6 1 31332 124471 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_7_2
S 31346 6 1 0 0 6 1 31332 124587 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12124
S 31347 6 1 0 0 6 1 31332 124597 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12126
S 31348 23 5 0 4 0 31355 624 124607 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 cumm
S 31349 7 3 0 0 15593 1 31348 5757 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 31350 7 3 0 0 15596 1 31348 5759 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 b
S 31351 1 3 0 0 6 1 31348 19100 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 m
S 31352 1 3 0 0 6 1 31348 19102 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 n
S 31353 1 3 0 0 6 1 31348 24501 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 k
S 31354 7 3 0 0 15599 1 31348 17822 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 c
S 31355 14 5 0 4 0 1 31348 124607 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 18211 6 0 0 0 0 0 0 0 0 0 0 0 0 415 0 624 0 0 0 0 cumm
F 31355 6 31349 31350 31351 31352 31353 31354
S 31356 6 1 0 0 6 1 31348 124421 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_2
S 31357 6 1 0 0 6 1 31348 124429 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_2
S 31358 6 1 0 0 6 1 31348 124437 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_2
S 31359 6 1 0 0 6 1 31348 124612 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12127
S 31360 6 1 0 0 6 1 31348 124455 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_4_2
S 31361 6 1 0 0 6 1 31348 124463 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_2
S 31362 6 1 0 0 6 1 31348 124471 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_7_2
S 31363 6 1 0 0 6 1 31348 124622 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12134
S 31364 6 1 0 0 6 1 31348 124489 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_8_2
S 31365 6 1 0 0 6 1 31348 124497 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_10_2
S 31366 6 1 0 0 6 1 31348 124506 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_11_1
S 31367 6 1 0 0 6 1 31348 124632 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12141
S 31368 23 5 0 4 0 31373 624 124642 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 cudotp
S 31369 7 3 0 0 15602 1 31368 124418 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v1
S 31370 7 3 0 0 15605 1 31368 124649 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v2
S 31371 1 3 0 0 6 1 31368 19102 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 n
S 31372 1 3 0 0 9 1 31368 124652 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 dotp
S 31373 14 5 0 4 0 1 31368 124642 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 18218 4 0 0 0 0 0 0 0 0 0 0 0 0 472 0 624 0 0 0 0 cudotp
F 31373 4 31369 31370 31371 31372
S 31374 6 1 0 0 6 1 31368 124421 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_2
S 31375 6 1 0 0 6 1 31368 124429 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_2
S 31376 6 1 0 0 6 1 31368 124437 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_2
S 31377 6 1 0 0 6 1 31368 124657 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12139
S 31378 6 1 0 0 6 1 31368 124455 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_4_2
S 31379 6 1 0 0 6 1 31368 124463 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_2
S 31380 6 1 0 0 6 1 31368 124471 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_7_2
S 31381 6 1 0 0 6 1 31368 124667 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12146
S 31382 23 5 0 4 0 31386 624 124677 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 cunorm2
S 31383 7 3 0 0 15608 1 31382 124418 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v1
S 31384 1 3 0 0 6 1 31382 19102 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 n
S 31385 1 3 0 0 9 1 31382 124685 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 norm
S 31386 14 5 0 4 0 1 31382 124677 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 18223 3 0 0 0 0 0 0 0 0 0 0 0 0 520 0 624 0 0 0 0 cunorm2
F 31386 3 31383 31384 31385
S 31387 6 1 0 0 6 1 31382 124421 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_2
S 31388 6 1 0 0 6 1 31382 124429 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_2
S 31389 6 1 0 0 6 1 31382 124437 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_2
S 31390 6 1 0 0 6 1 31382 124690 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12147
S 31391 23 5 0 4 0 31398 624 124700 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 cuaddvv
S 31392 7 3 0 0 15611 1 31391 124418 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v1
S 31393 7 3 0 0 15614 1 31391 124649 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v2
S 31394 1 3 0 0 9 1 31391 5757 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 31395 1 3 0 0 9 1 31391 5759 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 b
S 31396 7 3 0 0 15617 1 31391 124708 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v3
S 31397 1 3 0 0 6 1 31391 19102 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 n
S 31398 14 5 0 4 0 1 31391 124700 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 18227 6 0 0 0 0 0 0 0 0 0 0 0 0 568 0 624 0 0 0 0 cuaddvv
F 31398 6 31392 31393 31394 31395 31396 31397
S 31399 6 1 0 0 6 1 31391 124421 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_2
S 31400 6 1 0 0 6 1 31391 124429 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_2
S 31401 6 1 0 0 6 1 31391 124437 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_2
S 31402 6 1 0 0 6 1 31391 124711 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12151
S 31403 6 1 0 0 6 1 31391 124455 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_4_2
S 31404 6 1 0 0 6 1 31391 124463 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_2
S 31405 6 1 0 0 6 1 31391 124471 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_7_2
S 31406 6 1 0 0 6 1 31391 124721 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12158
S 31407 6 1 0 0 6 1 31391 124489 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_8_2
S 31408 6 1 0 0 6 1 31391 124497 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_10_2
S 31409 6 1 0 0 6 1 31391 124506 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_11_1
S 31410 6 1 0 0 6 1 31391 124731 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12165
S 31411 23 5 0 4 0 31416 624 124741 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 cusmulv
S 31412 7 3 0 0 15620 1 31411 124415 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v0
S 31413 1 3 0 0 9 1 31411 5757 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 31414 7 3 0 0 15623 1 31411 124418 20008004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v1
S 31415 1 3 0 0 6 1 31411 19102 8004 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 n
S 31416 14 5 0 4 0 1 31411 124741 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 18234 4 0 0 0 0 0 0 0 0 0 0 0 0 593 0 624 0 0 0 0 cusmulv
F 31416 4 31412 31413 31414 31415
S 31417 6 1 0 0 6 1 31411 124421 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_2
S 31418 6 1 0 0 6 1 31411 124429 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_2
S 31419 6 1 0 0 6 1 31411 124437 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_2
S 31420 6 1 0 0 6 1 31411 124749 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12163
S 31421 6 1 0 0 6 1 31411 124455 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_4_2
S 31422 6 1 0 0 6 1 31411 124463 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_2
S 31423 6 1 0 0 6 1 31411 124471 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_7_2
S 31424 6 1 0 0 6 1 31411 124759 40808006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_12170
A 18 2 0 0 0 6 632 0 0 0 18 0 0 0 0 0 0 0 0 0
A 67 1 0 0 0 56 682 0 0 0 0 0 0 0 0 0 0 0 0 0
A 70 1 0 0 0 62 684 0 0 0 0 0 0 0 0 0 0 0 0 0
A 86 1 0 0 0 92 722 0 0 0 0 0 0 0 0 0 0 0 0 0
A 87 2 0 0 0 6 779 0 0 0 87 0 0 0 0 0 0 0 0 0
A 89 2 0 0 0 6 783 0 0 0 89 0 0 0 0 0 0 0 0 0
A 90 2 0 0 0 6 785 0 0 0 90 0 0 0 0 0 0 0 0 0
A 92 2 0 0 0 6 792 0 0 0 92 0 0 0 0 0 0 0 0 0
A 95 2 0 0 0 6 791 0 0 0 95 0 0 0 0 0 0 0 0 0
A 96 2 0 0 0 6 789 0 0 0 96 0 0 0 0 0 0 0 0 0
A 100 2 0 0 0 6 788 0 0 0 100 0 0 0 0 0 0 0 0 0
A 101 2 0 0 0 6 801 0 0 0 101 0 0 0 0 0 0 0 0 0
A 103 2 0 0 0 6 790 0 0 0 103 0 0 0 0 0 0 0 0 0
A 120 2 0 0 0 6 778 0 0 0 120 0 0 0 0 0 0 0 0 0
A 2899 2 0 0 2793 16 11370 0 0 0 2899 0 0 0 0 0 0 0 0 0
A 11705 2 0 0 11661 12 31192 0 0 0 11705 0 0 0 0 0 0 0 0 0
A 11706 2 0 0 11588 12 31195 0 0 0 11706 0 0 0 0 0 0 0 0 0
A 11707 2 0 0 11450 12 31198 0 0 0 11707 0 0 0 0 0 0 0 0 0
A 11708 2 0 0 11191 12 31201 0 0 0 11708 0 0 0 0 0 0 0 0 0
A 11709 2 0 0 11193 12 31204 0 0 0 11709 0 0 0 0 0 0 0 0 0
A 11710 2 0 0 11453 12 31207 0 0 0 11710 0 0 0 0 0 0 0 0 0
A 11711 2 0 0 11326 12 31210 0 0 0 11711 0 0 0 0 0 0 0 0 0
A 11712 2 0 0 10932 12 31213 0 0 0 11712 0 0 0 0 0 0 0 0 0
A 11713 15 0 0 0 12 31189 11705 11714 0 0 0 0 0 0 0 0 0 0 0
A 11714 15 0 0 0 12 31189 11706 11715 0 0 0 0 0 0 0 0 0 0 0
A 11715 15 0 0 0 12 31189 11707 11716 0 0 0 0 0 0 0 0 0 0 0
A 11716 15 0 0 0 12 31189 11708 11717 0 0 0 0 0 0 0 0 0 0 0
A 11717 15 0 0 0 12 31189 11709 11718 0 0 0 0 0 0 0 0 0 0 0
A 11718 15 0 0 0 12 31189 11710 11719 0 0 0 0 0 0 0 0 0 0 0
A 11719 15 0 0 0 12 31189 11711 11720 0 0 0 0 0 0 0 0 0 0 0
A 11720 15 0 0 0 12 31189 11712 0 0 0 0 0 0 0 0 0 0 0 0
A 11721 15 0 0 0 15495 31189 11713 0 0 0 0 0 0 0 0 0 0 0 0
A 11722 1 0 13 11493 15492 31214 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11723 2 0 0 11328 12 31218 0 0 0 11723 0 0 0 0 0 0 0 0 0
A 11724 2 0 0 11198 12 31221 0 0 0 11724 0 0 0 0 0 0 0 0 0
A 11725 2 0 0 11458 12 31224 0 0 0 11725 0 0 0 0 0 0 0 0 0
A 11726 2 0 0 11665 12 31227 0 0 0 11726 0 0 0 0 0 0 0 0 0
A 11727 2 0 0 11200 12 31230 0 0 0 11727 0 0 0 0 0 0 0 0 0
A 11728 2 0 0 11460 12 31233 0 0 0 11728 0 0 0 0 0 0 0 0 0
A 11729 2 0 0 11595 12 31236 0 0 0 11729 0 0 0 0 0 0 0 0 0
A 11730 2 0 0 10941 12 31239 0 0 0 11730 0 0 0 0 0 0 0 0 0
A 11731 15 0 0 0 12 31215 11723 11732 0 0 0 0 0 0 0 0 0 0 0
A 11732 15 0 0 0 12 31215 11724 11733 0 0 0 0 0 0 0 0 0 0 0
A 11733 15 0 0 0 12 31215 11725 11734 0 0 0 0 0 0 0 0 0 0 0
A 11734 15 0 0 0 12 31215 11726 11735 0 0 0 0 0 0 0 0 0 0 0
A 11735 15 0 0 0 12 31215 11727 11736 0 0 0 0 0 0 0 0 0 0 0
A 11736 15 0 0 0 12 31215 11728 11737 0 0 0 0 0 0 0 0 0 0 0
A 11737 15 0 0 0 12 31215 11729 11738 0 0 0 0 0 0 0 0 0 0 0
A 11738 15 0 0 0 12 31215 11730 0 0 0 0 0 0 0 0 0 0 0 0
A 11739 15 0 0 0 15501 31215 11731 0 0 0 0 0 0 0 0 0 0 0 0
A 11740 1 0 13 11684 15498 31240 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11741 2 0 0 11333 9 31242 0 0 0 11741 0 0 0 0 0 0 0 0 0
A 11745 1 0 3 11346 15513 31249 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11746 10 0 0 11074 6 11745 4 0 0 0 0 0 0 0 0 0 0 0 0
X 1 100
A 11747 10 0 0 11746 6 11745 7 0 0 0 0 0 0 0 0 0 0 0 0
X 1 87
A 11748 4 0 0 11099 6 11747 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11749 4 0 0 11353 6 11746 0 11748 0 0 0 0 1 0 0 0 0 0 0
A 11750 10 0 0 11747 6 11745 16 0 0 0 0 0 0 0 0 0 0 0 0
X 1 103
A 11751 10 0 0 11750 6 11745 19 0 0 0 0 0 0 0 0 0 0 0 0
X 1 95
A 11752 4 0 0 11488 6 11751 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11753 4 0 0 10807 6 11750 0 11752 0 0 0 0 1 0 0 0 0 0 0
A 11754 10 0 0 11751 6 11745 10 0 0 0 0 0 0 0 0 0 0 0 0
X 1 90
A 11755 10 0 0 11754 6 11745 22 0 0 0 0 0 0 0 0 0 0 0 0
X 1 92
A 11756 10 0 0 11755 6 11745 13 0 0 0 0 0 0 0 0 0 0 0 0
X 1 120
A 11757 10 0 0 11756 6 11745 1 0 0 0 0 0 0 0 0 0 0 0 0
X 1 18
A 11759 1 0 1 11673 15519 31255 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11760 10 0 0 11337 6 11759 4 0 0 0 0 0 0 0 0 0 0 0 0
X 1 100
A 11761 10 0 0 11760 6 11759 7 0 0 0 0 0 0 0 0 0 0 0 0
X 1 87
A 11762 4 0 0 11362 6 11761 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11763 4 0 0 11384 6 11760 0 11762 0 0 0 0 1 0 0 0 0 0 0
A 11764 10 0 0 11761 6 11759 10 0 0 0 0 0 0 0 0 0 0 0 0
X 1 90
A 11765 10 0 0 11764 6 11759 13 0 0 0 0 0 0 0 0 0 0 0 0
X 1 120
A 11766 10 0 0 11765 6 11759 1 0 0 0 0 0 0 0 0 0 0 0 0
X 1 18
A 11768 1 0 1 11654 15525 31261 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11769 10 0 0 11599 6 11768 4 0 0 0 0 0 0 0 0 0 0 0 0
X 1 100
A 11770 10 0 0 11769 6 11768 7 0 0 0 0 0 0 0 0 0 0 0 0
X 1 87
A 11771 4 0 0 10026 6 11770 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11772 4 0 0 11175 6 11769 0 11771 0 0 0 0 1 0 0 0 0 0 0
A 11773 10 0 0 11770 6 11768 10 0 0 0 0 0 0 0 0 0 0 0 0
X 1 90
A 11774 10 0 0 11773 6 11768 13 0 0 0 0 0 0 0 0 0 0 0 0
X 1 120
A 11775 10 0 0 11774 6 11768 1 0 0 0 0 0 0 0 0 0 0 0 0
X 1 18
A 11778 1 0 3 11767 15531 31268 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11779 10 0 0 11344 6 11778 4 0 0 0 0 0 0 0 0 0 0 0 0
X 1 100
A 11780 10 0 0 11779 6 11778 7 0 0 0 0 0 0 0 0 0 0 0 0
X 1 87
A 11781 4 0 0 11680 6 11780 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11782 4 0 0 11548 6 11779 0 11781 0 0 0 0 1 0 0 0 0 0 0
A 11783 10 0 0 11780 6 11778 16 0 0 0 0 0 0 0 0 0 0 0 0
X 1 103
A 11784 10 0 0 11783 6 11778 19 0 0 0 0 0 0 0 0 0 0 0 0
X 1 95
A 11785 4 0 0 11722 6 11784 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11786 4 0 0 11281 6 11783 0 11785 0 0 0 0 1 0 0 0 0 0 0
A 11787 10 0 0 11784 6 11778 10 0 0 0 0 0 0 0 0 0 0 0 0
X 1 90
A 11788 10 0 0 11787 6 11778 22 0 0 0 0 0 0 0 0 0 0 0 0
X 1 92
A 11789 10 0 0 11788 6 11778 13 0 0 0 0 0 0 0 0 0 0 0 0
X 1 120
A 11790 10 0 0 11789 6 11778 1 0 0 0 0 0 0 0 0 0 0 0 0
X 1 18
A 11792 1 0 1 11638 15537 31275 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11793 10 0 0 11217 6 11792 4 0 0 0 0 0 0 0 0 0 0 0 0
X 1 100
A 11794 10 0 0 11793 6 11792 7 0 0 0 0 0 0 0 0 0 0 0 0
X 1 87
A 11795 4 0 0 11616 6 11794 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11796 4 0 0 11574 6 11793 0 11795 0 0 0 0 1 0 0 0 0 0 0
A 11797 10 0 0 11794 6 11792 10 0 0 0 0 0 0 0 0 0 0 0 0
X 1 90
A 11798 10 0 0 11797 6 11792 13 0 0 0 0 0 0 0 0 0 0 0 0
X 1 120
A 11799 10 0 0 11798 6 11792 1 0 0 0 0 0 0 0 0 0 0 0 0
X 1 18
A 11801 1 0 1 11555 15543 31281 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11802 10 0 0 11089 6 11801 4 0 0 0 0 0 0 0 0 0 0 0 0
X 1 100
A 11803 10 0 0 11802 6 11801 7 0 0 0 0 0 0 0 0 0 0 0 0
X 1 87
A 11804 4 0 0 10987 6 11803 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11805 4 0 0 10967 6 11802 0 11804 0 0 0 0 1 0 0 0 0 0 0
A 11806 10 0 0 11803 6 11801 10 0 0 0 0 0 0 0 0 0 0 0 0
X 1 90
A 11807 10 0 0 11806 6 11801 13 0 0 0 0 0 0 0 0 0 0 0 0
X 1 120
A 11808 10 0 0 11807 6 11801 1 0 0 0 0 0 0 0 0 0 0 0 0
X 1 18
A 11810 1 0 1 11033 15549 31287 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11811 10 0 0 11606 6 11810 4 0 0 0 0 0 0 0 0 0 0 0 0
X 1 100
A 11812 10 0 0 11811 6 11810 7 0 0 0 0 0 0 0 0 0 0 0 0
X 1 87
A 11813 4 0 0 11617 6 11812 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11814 4 0 0 11622 6 11811 0 11813 0 0 0 0 1 0 0 0 0 0 0
A 11815 10 0 0 11812 6 11810 10 0 0 0 0 0 0 0 0 0 0 0 0
X 1 90
A 11816 10 0 0 11815 6 11810 13 0 0 0 0 0 0 0 0 0 0 0 0
X 1 120
A 11817 10 0 0 11816 6 11810 1 0 0 0 0 0 0 0 0 0 0 0 0
X 1 18
A 11820 1 0 3 11372 15555 31294 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11821 10 0 0 11484 6 11820 4 0 0 0 0 0 0 0 0 0 0 0 0
X 1 100
A 11822 10 0 0 11821 6 11820 7 0 0 0 0 0 0 0 0 0 0 0 0
X 1 87
A 11823 4 0 0 11116 6 11822 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11824 4 0 0 11498 6 11821 0 11823 0 0 0 0 1 0 0 0 0 0 0
A 11825 10 0 0 11822 6 11820 16 0 0 0 0 0 0 0 0 0 0 0 0
X 1 103
A 11826 10 0 0 11825 6 11820 19 0 0 0 0 0 0 0 0 0 0 0 0
X 1 95
A 11827 4 0 0 11379 6 11826 0 3 0 0 0 0 2 0 0 0 0 0 0
A 11828 4 0 0 11412 6 11825 0 11827 0 0 0 0 1 0 0 0 0 0 0
A 11829 10 0 0 11826 6 11820 10 0 0 0 0 0 0 0 0 0 0 0 0
X 1 90
A 11830 10 0 0 11829 6 11820 22 0 0 0 0 0 0 0 0 0 0 0 0
X 1 92
A 11831 10 0 0 11830 6 11820 13 0 0 0 0 0 0 0 0 0 0 0 0
X 1 120
A 11832 10 0 0 11831 6 11820 1 0 0 0 0 0 0 0 0 0 0 0 0
X 1 18
A 11833 1 0 0 11323 6 31317 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11834 1 0 0 11192 6 31315 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11835 1 0 0 11063 6 31318 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11836 1 0 0 11709 6 31316 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11837 1 0 0 11663 6 31321 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11838 1 0 0 11710 6 31319 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11839 1 0 0 11711 6 31322 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11840 1 0 0 11662 6 31320 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11841 1 0 0 11712 6 31325 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11842 1 0 0 11195 6 31323 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11843 1 0 0 11590 6 31326 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11844 1 0 0 10497 6 31324 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11845 1 0 0 11456 6 31329 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11846 1 0 0 11591 6 31327 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11847 1 0 0 11723 6 31330 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11848 1 0 0 11592 6 31328 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11849 1 0 0 11707 6 31310 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11850 1 0 0 11194 6 31331 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11851 1 0 0 10935 6 31341 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11852 1 0 0 11726 6 31339 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11853 1 0 0 11727 6 31342 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11854 1 0 0 11331 6 31340 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11855 1 0 0 11728 6 31345 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11856 1 0 0 11459 6 31343 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11857 1 0 0 11593 6 31346 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11858 1 0 0 11457 6 31344 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11859 1 0 0 9404 6 31335 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11860 1 0 0 11594 6 31347 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11861 1 0 0 11462 6 31358 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11862 1 0 0 11667 6 31356 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11863 1 0 0 11465 6 31359 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11864 1 0 0 11464 6 31357 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11865 1 0 0 11207 6 31362 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11866 1 0 0 11203 6 31360 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11867 1 0 0 10504 6 31363 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11868 1 0 0 11206 6 31361 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11869 1 0 0 11336 6 31366 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11870 1 0 0 11757 6 31364 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11871 1 0 0 11596 6 31367 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11872 1 0 0 11335 6 31365 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11873 1 0 0 11209 6 31376 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11874 1 0 0 11669 6 31374 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11875 1 0 0 11210 6 31377 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11876 1 0 0 11470 6 31375 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11877 1 0 0 11339 6 31380 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11878 1 0 0 11766 6 31378 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11879 1 0 0 10951 6 31381 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11880 1 0 0 11338 6 31379 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11881 1 0 0 11601 6 31389 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11882 1 0 0 11775 6 31387 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11883 1 0 0 11670 6 31390 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11884 1 0 0 11600 6 31388 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11885 1 0 0 11475 6 31401 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11886 1 0 0 10139 6 31399 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11887 1 0 0 11215 6 31402 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11888 1 0 0 11474 6 31400 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11889 1 0 0 11085 6 31405 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11890 1 0 0 10517 6 31403 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11891 1 0 0 11745 6 31406 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11892 1 0 0 10518 6 31404 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11893 1 0 0 11603 6 31409 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11894 1 0 0 11672 6 31407 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11895 1 0 0 11604 6 31410 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11896 1 0 0 11759 6 31408 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11897 1 0 0 10299 6 31419 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11898 1 0 0 11219 6 31417 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11899 1 0 0 11808 6 31420 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11900 1 0 0 10454 6 31418 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11901 1 0 0 11351 6 31423 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11902 1 0 0 11350 6 31421 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11903 1 0 0 11674 6 31424 0 0 0 0 0 0 0 0 0 0 0 0 0
A 11904 1 0 0 11090 6 31422 0 0 0 0 0 0 0 0 0 0 0 0 0
Z
J 149 1 1
V 67 56 7 0
S 0 56 0 0 0
A 0 6 0 0 1 2 0
J 150 1 1
V 70 62 7 0
S 0 62 0 0 0
A 0 6 0 0 1 2 0
J 31 1 1
V 86 92 7 0
S 0 92 0 0 0
A 0 74 0 0 1 67 0
J 33 1 1
V 11722 15492 7 0
R 0 15495 0 0
A 0 12 0 0 1 11705 1
A 0 12 0 0 1 11706 1
A 0 12 0 0 1 11707 1
A 0 12 0 0 1 11708 1
A 0 12 0 0 1 11709 1
A 0 12 0 0 1 11710 1
A 0 12 0 0 1 11711 1
A 0 12 0 0 1 11712 0
J 37 1 1
V 11740 15498 7 0
R 0 15501 0 0
A 0 12 0 0 1 11723 1
A 0 12 0 0 1 11724 1
A 0 12 0 0 1 11725 1
A 0 12 0 0 1 11726 1
A 0 12 0 0 1 11727 1
A 0 12 0 0 1 11728 1
A 0 12 0 0 1 11729 1
A 0 12 0 0 1 11730 0
T 11371 7026 0 3 0 0
A 11438 16 0 0 1 2899 1
A 11439 16 0 0 1 2899 1
A 11440 16 0 0 1 2899 1
A 11441 16 0 0 1 2899 0
T 11442 7095 0 3 0 0
A 11509 16 0 0 1 2899 1
A 11510 16 0 0 1 2899 1
A 11511 16 0 0 1 2899 1
A 11512 16 0 0 1 2899 0
T 11513 7164 0 3 0 0
A 11580 16 0 0 1 2899 1
A 11581 16 0 0 1 2899 1
A 11582 16 0 0 1 2899 1
A 11583 16 0 0 1 2899 0
T 11584 7233 0 3 0 0
A 11651 16 0 0 1 2899 1
A 11652 16 0 0 1 2899 1
A 11653 16 0 0 1 2899 1
A 11654 16 0 0 1 2899 0
T 11655 7302 0 3 0 0
A 11728 16 0 0 1 2899 1
A 11729 16 0 0 1 2899 1
A 11730 16 0 0 1 2899 1
A 11731 16 0 0 1 2899 0
T 11732 7377 0 3 0 0
A 11805 16 0 0 1 2899 1
A 11806 16 0 0 1 2899 1
A 11807 16 0 0 1 2899 1
A 11808 16 0 0 1 2899 0
T 31243 15504 0 3 0 0
T 31272 7164 0 3 0 0
A 11580 16 0 0 1 2899 1
A 11581 16 0 0 1 2899 1
A 11582 16 0 0 1 2899 1
A 11583 16 0 0 1 2899 0
Z
