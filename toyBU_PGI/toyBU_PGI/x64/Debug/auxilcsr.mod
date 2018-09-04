V29 :0x14 auxilcsr
61 F:\Archive\toy\Depletion\toyBU_PGI\toyBU_PGI\AuxiliaryCSR.f90 S624 0
08/20/2018  17:44:24
use iso_c_binding public 0 indirect
use cusparse public 0 indirect
use csrmatrix public 0 direct
use pgi_acc_common private
enduse
D 56 24 644 8 643 7
D 62 24 646 8 645 7
D 74 24 644 8 643 7
D 92 24 717 8 716 7
D 7026 24 11370 1200 11369 7
D 7095 24 11442 1200 11440 7
D 7164 24 11513 1200 11511 7
D 7233 24 11584 1200 11582 7
D 7302 24 11655 1288 11653 7
D 7377 24 11732 1288 11730 7
D 7470 21 9 2 2900 2906 1 1 0 0 1
 3 2901 3 3 2901 2902
 3 2903 2904 3 2903 2905
D 7473 21 9 2 2907 2913 1 1 0 0 1
 3 2908 3 3 2908 2909
 3 2910 2911 3 2910 2912
D 7476 21 12 2 2914 2920 1 1 0 0 1
 3 2915 3 3 2915 2916
 3 2917 2918 3 2917 2919
D 7479 21 9 2 2921 2927 1 1 0 0 1
 3 2922 3 3 2922 2923
 3 2924 2925 3 2924 2926
D 7482 21 9 2 2928 2934 1 1 0 0 1
 3 2929 3 3 2929 2930
 3 2931 2932 3 2931 2933
D 7485 21 12 2 2935 2941 1 1 0 0 1
 3 2936 3 3 2936 2937
 3 2938 2939 3 2938 2940
D 7488 21 12 2 2942 2948 1 1 0 0 1
 3 2943 3 3 2943 2944
 3 2945 2946 3 2945 2947
S 624 24 0 0 0 9 1 0 5012 10005 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 auxilcsr
R 643 25 6 iso_c_binding c_ptr
R 644 5 7 iso_c_binding val c_ptr
R 645 25 8 iso_c_binding c_funptr
R 646 5 9 iso_c_binding val c_funptr
R 679 6 42 iso_c_binding c_null_ptr$ac
R 681 6 44 iso_c_binding c_null_funptr$ac
R 682 26 45 iso_c_binding ==
R 684 26 47 iso_c_binding !=
R 716 25 5 pgi_acc_common c_devptr
R 717 5 6 pgi_acc_common cptr c_devptr
R 719 6 8 pgi_acc_common c_null_devptr$ac
R 723 26 12 pgi_acc_common =
S 11368 3 0 0 0 16 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16
R 11369 25 1 csrmatrix csr_float
R 11370 5 2 csrmatrix csrval csr_float
R 11372 5 4 csrmatrix csrval$sd csr_float
R 11373 5 5 csrmatrix csrval$p csr_float
R 11374 5 6 csrmatrix csrval$o csr_float
R 11376 5 8 csrmatrix csrrowptr csr_float
R 11378 5 10 csrmatrix csrrowptr$sd csr_float
R 11379 5 11 csrmatrix csrrowptr$p csr_float
R 11380 5 12 csrmatrix csrrowptr$o csr_float
R 11382 5 14 csrmatrix csrcolidx csr_float
R 11384 5 16 csrmatrix csrcolidx$sd csr_float
R 11385 5 17 csrmatrix csrcolidx$p csr_float
R 11386 5 18 csrmatrix csrcolidx$o csr_float
R 11388 5 20 csrmatrix bufval csr_float
R 11391 5 23 csrmatrix bufval$sd csr_float
R 11392 5 24 csrmatrix bufval$p csr_float
R 11393 5 25 csrmatrix bufval$o csr_float
R 11395 5 27 csrmatrix bufrowptr csr_float
R 11398 5 30 csrmatrix bufrowptr$sd csr_float
R 11399 5 31 csrmatrix bufrowptr$p csr_float
R 11400 5 32 csrmatrix bufrowptr$o csr_float
R 11402 5 34 csrmatrix bufcolidx csr_float
R 11405 5 37 csrmatrix bufcolidx$sd csr_float
R 11406 5 38 csrmatrix bufcolidx$p csr_float
R 11407 5 39 csrmatrix bufcolidx$o csr_float
R 11409 5 41 csrmatrix d_csrval csr_float
R 11411 5 43 csrmatrix d_csrval$sd csr_float
R 11412 5 44 csrmatrix d_csrval$p csr_float
R 11413 5 45 csrmatrix d_csrval$o csr_float
R 11415 5 47 csrmatrix d_csrrowptr csr_float
R 11417 5 49 csrmatrix d_csrrowptr$sd csr_float
R 11418 5 50 csrmatrix d_csrrowptr$p csr_float
R 11419 5 51 csrmatrix d_csrrowptr$o csr_float
R 11421 5 53 csrmatrix d_csrcolidx csr_float
R 11423 5 55 csrmatrix d_csrcolidx$sd csr_float
R 11424 5 56 csrmatrix d_csrcolidx$p csr_float
R 11425 5 57 csrmatrix d_csrcolidx$o csr_float
R 11427 5 59 csrmatrix info csr_float
R 11428 5 60 csrmatrix infolu csr_float
R 11429 5 61 csrmatrix descr csr_float
R 11430 5 62 csrmatrix descrlu csr_float
R 11431 5 63 csrmatrix nr csr_float
R 11432 5 64 csrmatrix nc csr_float
R 11433 5 65 csrmatrix nnz csr_float
R 11434 5 66 csrmatrix nnzomp csr_float
R 11435 5 67 csrmatrix nthread csr_float
R 11436 5 68 csrmatrix lalloc csr_float
R 11437 5 69 csrmatrix lfinalized csr_float
R 11438 5 70 csrmatrix ldevice csr_float
R 11439 5 71 csrmatrix lparallel csr_float
R 11440 25 72 csrmatrix csr_float_complex
R 11442 5 74 csrmatrix csrval csr_float_complex
R 11443 5 75 csrmatrix csrval$sd csr_float_complex
R 11444 5 76 csrmatrix csrval$p csr_float_complex
R 11445 5 77 csrmatrix csrval$o csr_float_complex
R 11448 5 80 csrmatrix csrrowptr csr_float_complex
R 11449 5 81 csrmatrix csrrowptr$sd csr_float_complex
R 11450 5 82 csrmatrix csrrowptr$p csr_float_complex
R 11451 5 83 csrmatrix csrrowptr$o csr_float_complex
R 11454 5 86 csrmatrix csrcolidx csr_float_complex
R 11455 5 87 csrmatrix csrcolidx$sd csr_float_complex
R 11456 5 88 csrmatrix csrcolidx$p csr_float_complex
R 11457 5 89 csrmatrix csrcolidx$o csr_float_complex
R 11461 5 93 csrmatrix bufval csr_float_complex
R 11462 5 94 csrmatrix bufval$sd csr_float_complex
R 11463 5 95 csrmatrix bufval$p csr_float_complex
R 11464 5 96 csrmatrix bufval$o csr_float_complex
R 11468 5 100 csrmatrix bufrowptr csr_float_complex
R 11469 5 101 csrmatrix bufrowptr$sd csr_float_complex
R 11470 5 102 csrmatrix bufrowptr$p csr_float_complex
R 11471 5 103 csrmatrix bufrowptr$o csr_float_complex
R 11475 5 107 csrmatrix bufcolidx csr_float_complex
R 11476 5 108 csrmatrix bufcolidx$sd csr_float_complex
R 11477 5 109 csrmatrix bufcolidx$p csr_float_complex
R 11478 5 110 csrmatrix bufcolidx$o csr_float_complex
R 11481 5 113 csrmatrix d_csrval csr_float_complex
R 11482 5 114 csrmatrix d_csrval$sd csr_float_complex
R 11483 5 115 csrmatrix d_csrval$p csr_float_complex
R 11484 5 116 csrmatrix d_csrval$o csr_float_complex
R 11487 5 119 csrmatrix d_csrrowptr csr_float_complex
R 11488 5 120 csrmatrix d_csrrowptr$sd csr_float_complex
R 11489 5 121 csrmatrix d_csrrowptr$p csr_float_complex
R 11490 5 122 csrmatrix d_csrrowptr$o csr_float_complex
R 11493 5 125 csrmatrix d_csrcolidx csr_float_complex
R 11494 5 126 csrmatrix d_csrcolidx$sd csr_float_complex
R 11495 5 127 csrmatrix d_csrcolidx$p csr_float_complex
R 11496 5 128 csrmatrix d_csrcolidx$o csr_float_complex
R 11498 5 130 csrmatrix info csr_float_complex
R 11499 5 131 csrmatrix infolu csr_float_complex
R 11500 5 132 csrmatrix descr csr_float_complex
R 11501 5 133 csrmatrix descrlu csr_float_complex
R 11502 5 134 csrmatrix nr csr_float_complex
R 11503 5 135 csrmatrix nc csr_float_complex
R 11504 5 136 csrmatrix nnz csr_float_complex
R 11505 5 137 csrmatrix nnzomp csr_float_complex
R 11506 5 138 csrmatrix nthread csr_float_complex
R 11507 5 139 csrmatrix lalloc csr_float_complex
R 11508 5 140 csrmatrix lfinalized csr_float_complex
R 11509 5 141 csrmatrix ldevice csr_float_complex
R 11510 5 142 csrmatrix lparallel csr_float_complex
R 11511 25 143 csrmatrix csr_double
R 11513 5 145 csrmatrix csrval csr_double
R 11514 5 146 csrmatrix csrval$sd csr_double
R 11515 5 147 csrmatrix csrval$p csr_double
R 11516 5 148 csrmatrix csrval$o csr_double
R 11519 5 151 csrmatrix csrrowptr csr_double
R 11520 5 152 csrmatrix csrrowptr$sd csr_double
R 11521 5 153 csrmatrix csrrowptr$p csr_double
R 11522 5 154 csrmatrix csrrowptr$o csr_double
R 11525 5 157 csrmatrix csrcolidx csr_double
R 11526 5 158 csrmatrix csrcolidx$sd csr_double
R 11527 5 159 csrmatrix csrcolidx$p csr_double
R 11528 5 160 csrmatrix csrcolidx$o csr_double
R 11532 5 164 csrmatrix bufval csr_double
R 11533 5 165 csrmatrix bufval$sd csr_double
R 11534 5 166 csrmatrix bufval$p csr_double
R 11535 5 167 csrmatrix bufval$o csr_double
R 11539 5 171 csrmatrix bufrowptr csr_double
R 11540 5 172 csrmatrix bufrowptr$sd csr_double
R 11541 5 173 csrmatrix bufrowptr$p csr_double
R 11542 5 174 csrmatrix bufrowptr$o csr_double
R 11546 5 178 csrmatrix bufcolidx csr_double
R 11547 5 179 csrmatrix bufcolidx$sd csr_double
R 11548 5 180 csrmatrix bufcolidx$p csr_double
R 11549 5 181 csrmatrix bufcolidx$o csr_double
R 11552 5 184 csrmatrix d_csrval csr_double
R 11553 5 185 csrmatrix d_csrval$sd csr_double
R 11554 5 186 csrmatrix d_csrval$p csr_double
R 11555 5 187 csrmatrix d_csrval$o csr_double
R 11558 5 190 csrmatrix d_csrrowptr csr_double
R 11559 5 191 csrmatrix d_csrrowptr$sd csr_double
R 11560 5 192 csrmatrix d_csrrowptr$p csr_double
R 11561 5 193 csrmatrix d_csrrowptr$o csr_double
R 11564 5 196 csrmatrix d_csrcolidx csr_double
R 11565 5 197 csrmatrix d_csrcolidx$sd csr_double
R 11566 5 198 csrmatrix d_csrcolidx$p csr_double
R 11567 5 199 csrmatrix d_csrcolidx$o csr_double
R 11569 5 201 csrmatrix info csr_double
R 11570 5 202 csrmatrix infolu csr_double
R 11571 5 203 csrmatrix descr csr_double
R 11572 5 204 csrmatrix descrlu csr_double
R 11573 5 205 csrmatrix nr csr_double
R 11574 5 206 csrmatrix nc csr_double
R 11575 5 207 csrmatrix nnz csr_double
R 11576 5 208 csrmatrix nnzomp csr_double
R 11577 5 209 csrmatrix nthread csr_double
R 11578 5 210 csrmatrix lalloc csr_double
R 11579 5 211 csrmatrix lfinalized csr_double
R 11580 5 212 csrmatrix ldevice csr_double
R 11581 5 213 csrmatrix lparallel csr_double
R 11582 25 214 csrmatrix csr_double_complex
R 11584 5 216 csrmatrix csrval csr_double_complex
R 11585 5 217 csrmatrix csrval$sd csr_double_complex
R 11586 5 218 csrmatrix csrval$p csr_double_complex
R 11587 5 219 csrmatrix csrval$o csr_double_complex
R 11590 5 222 csrmatrix csrrowptr csr_double_complex
R 11591 5 223 csrmatrix csrrowptr$sd csr_double_complex
R 11592 5 224 csrmatrix csrrowptr$p csr_double_complex
R 11593 5 225 csrmatrix csrrowptr$o csr_double_complex
R 11596 5 228 csrmatrix csrcolidx csr_double_complex
R 11597 5 229 csrmatrix csrcolidx$sd csr_double_complex
R 11598 5 230 csrmatrix csrcolidx$p csr_double_complex
R 11599 5 231 csrmatrix csrcolidx$o csr_double_complex
R 11603 5 235 csrmatrix bufval csr_double_complex
R 11604 5 236 csrmatrix bufval$sd csr_double_complex
R 11605 5 237 csrmatrix bufval$p csr_double_complex
R 11606 5 238 csrmatrix bufval$o csr_double_complex
R 11610 5 242 csrmatrix bufrowptr csr_double_complex
R 11611 5 243 csrmatrix bufrowptr$sd csr_double_complex
R 11612 5 244 csrmatrix bufrowptr$p csr_double_complex
R 11613 5 245 csrmatrix bufrowptr$o csr_double_complex
R 11617 5 249 csrmatrix bufcolidx csr_double_complex
R 11618 5 250 csrmatrix bufcolidx$sd csr_double_complex
R 11619 5 251 csrmatrix bufcolidx$p csr_double_complex
R 11620 5 252 csrmatrix bufcolidx$o csr_double_complex
R 11623 5 255 csrmatrix d_csrval csr_double_complex
R 11624 5 256 csrmatrix d_csrval$sd csr_double_complex
R 11625 5 257 csrmatrix d_csrval$p csr_double_complex
R 11626 5 258 csrmatrix d_csrval$o csr_double_complex
R 11629 5 261 csrmatrix d_csrrowptr csr_double_complex
R 11630 5 262 csrmatrix d_csrrowptr$sd csr_double_complex
R 11631 5 263 csrmatrix d_csrrowptr$p csr_double_complex
R 11632 5 264 csrmatrix d_csrrowptr$o csr_double_complex
R 11635 5 267 csrmatrix d_csrcolidx csr_double_complex
R 11636 5 268 csrmatrix d_csrcolidx$sd csr_double_complex
R 11637 5 269 csrmatrix d_csrcolidx$p csr_double_complex
R 11638 5 270 csrmatrix d_csrcolidx$o csr_double_complex
R 11640 5 272 csrmatrix info csr_double_complex
R 11641 5 273 csrmatrix infolu csr_double_complex
R 11642 5 274 csrmatrix descr csr_double_complex
R 11643 5 275 csrmatrix descrlu csr_double_complex
R 11644 5 276 csrmatrix nr csr_double_complex
R 11645 5 277 csrmatrix nc csr_double_complex
R 11646 5 278 csrmatrix nnz csr_double_complex
R 11647 5 279 csrmatrix nnzomp csr_double_complex
R 11648 5 280 csrmatrix nthread csr_double_complex
R 11649 5 281 csrmatrix lalloc csr_double_complex
R 11650 5 282 csrmatrix lfinalized csr_double_complex
R 11651 5 283 csrmatrix ldevice csr_double_complex
R 11652 5 284 csrmatrix lparallel csr_double_complex
R 11653 25 285 csrmatrix csr_mixed
R 11655 5 287 csrmatrix csrval csr_mixed
R 11656 5 288 csrmatrix csrval$sd csr_mixed
R 11657 5 289 csrmatrix csrval$p csr_mixed
R 11658 5 290 csrmatrix csrval$o csr_mixed
R 11661 5 293 csrmatrix csrrowptr csr_mixed
R 11662 5 294 csrmatrix csrrowptr$sd csr_mixed
R 11663 5 295 csrmatrix csrrowptr$p csr_mixed
R 11664 5 296 csrmatrix csrrowptr$o csr_mixed
R 11667 5 299 csrmatrix csrcolidx csr_mixed
R 11668 5 300 csrmatrix csrcolidx$sd csr_mixed
R 11669 5 301 csrmatrix csrcolidx$p csr_mixed
R 11670 5 302 csrmatrix csrcolidx$o csr_mixed
R 11674 5 306 csrmatrix bufval csr_mixed
R 11675 5 307 csrmatrix bufval$sd csr_mixed
R 11676 5 308 csrmatrix bufval$p csr_mixed
R 11677 5 309 csrmatrix bufval$o csr_mixed
R 11681 5 313 csrmatrix bufrowptr csr_mixed
R 11682 5 314 csrmatrix bufrowptr$sd csr_mixed
R 11683 5 315 csrmatrix bufrowptr$p csr_mixed
R 11684 5 316 csrmatrix bufrowptr$o csr_mixed
R 11688 5 320 csrmatrix bufcolidx csr_mixed
R 11689 5 321 csrmatrix bufcolidx$sd csr_mixed
R 11690 5 322 csrmatrix bufcolidx$p csr_mixed
R 11691 5 323 csrmatrix bufcolidx$o csr_mixed
R 11694 5 326 csrmatrix d_csrval csr_mixed
R 11695 5 327 csrmatrix d_csrval$sd csr_mixed
R 11696 5 328 csrmatrix d_csrval$p csr_mixed
R 11697 5 329 csrmatrix d_csrval$o csr_mixed
R 11699 5 331 csrmatrix d_csrval8 csr_mixed
R 11701 5 333 csrmatrix d_csrval8$sd csr_mixed
R 11702 5 334 csrmatrix d_csrval8$p csr_mixed
R 11703 5 335 csrmatrix d_csrval8$o csr_mixed
R 11706 5 338 csrmatrix d_csrrowptr csr_mixed
R 11707 5 339 csrmatrix d_csrrowptr$sd csr_mixed
R 11708 5 340 csrmatrix d_csrrowptr$p csr_mixed
R 11709 5 341 csrmatrix d_csrrowptr$o csr_mixed
R 11712 5 344 csrmatrix d_csrcolidx csr_mixed
R 11713 5 345 csrmatrix d_csrcolidx$sd csr_mixed
R 11714 5 346 csrmatrix d_csrcolidx$p csr_mixed
R 11715 5 347 csrmatrix d_csrcolidx$o csr_mixed
R 11717 5 349 csrmatrix info csr_mixed
R 11718 5 350 csrmatrix infolu csr_mixed
R 11719 5 351 csrmatrix descr csr_mixed
R 11720 5 352 csrmatrix descrlu csr_mixed
R 11721 5 353 csrmatrix nr csr_mixed
R 11722 5 354 csrmatrix nc csr_mixed
R 11723 5 355 csrmatrix nnz csr_mixed
R 11724 5 356 csrmatrix nnzomp csr_mixed
R 11725 5 357 csrmatrix nthread csr_mixed
R 11726 5 358 csrmatrix lalloc csr_mixed
R 11727 5 359 csrmatrix lfinalized csr_mixed
R 11728 5 360 csrmatrix ldevice csr_mixed
R 11729 5 361 csrmatrix lparallel csr_mixed
R 11730 25 362 csrmatrix csr_mixed_complex
R 11732 5 364 csrmatrix csrval csr_mixed_complex
R 11733 5 365 csrmatrix csrval$sd csr_mixed_complex
R 11734 5 366 csrmatrix csrval$p csr_mixed_complex
R 11735 5 367 csrmatrix csrval$o csr_mixed_complex
R 11738 5 370 csrmatrix csrrowptr csr_mixed_complex
R 11739 5 371 csrmatrix csrrowptr$sd csr_mixed_complex
R 11740 5 372 csrmatrix csrrowptr$p csr_mixed_complex
R 11741 5 373 csrmatrix csrrowptr$o csr_mixed_complex
R 11744 5 376 csrmatrix csrcolidx csr_mixed_complex
R 11745 5 377 csrmatrix csrcolidx$sd csr_mixed_complex
R 11746 5 378 csrmatrix csrcolidx$p csr_mixed_complex
R 11747 5 379 csrmatrix csrcolidx$o csr_mixed_complex
R 11751 5 383 csrmatrix bufval csr_mixed_complex
R 11752 5 384 csrmatrix bufval$sd csr_mixed_complex
R 11753 5 385 csrmatrix bufval$p csr_mixed_complex
R 11754 5 386 csrmatrix bufval$o csr_mixed_complex
R 11758 5 390 csrmatrix bufrowptr csr_mixed_complex
R 11759 5 391 csrmatrix bufrowptr$sd csr_mixed_complex
R 11760 5 392 csrmatrix bufrowptr$p csr_mixed_complex
R 11761 5 393 csrmatrix bufrowptr$o csr_mixed_complex
R 11765 5 397 csrmatrix bufcolidx csr_mixed_complex
R 11766 5 398 csrmatrix bufcolidx$sd csr_mixed_complex
R 11767 5 399 csrmatrix bufcolidx$p csr_mixed_complex
R 11768 5 400 csrmatrix bufcolidx$o csr_mixed_complex
R 11771 5 403 csrmatrix d_csrval csr_mixed_complex
R 11772 5 404 csrmatrix d_csrval$sd csr_mixed_complex
R 11773 5 405 csrmatrix d_csrval$p csr_mixed_complex
R 11774 5 406 csrmatrix d_csrval$o csr_mixed_complex
R 11777 5 409 csrmatrix d_csrval8 csr_mixed_complex
R 11778 5 410 csrmatrix d_csrval8$sd csr_mixed_complex
R 11779 5 411 csrmatrix d_csrval8$p csr_mixed_complex
R 11780 5 412 csrmatrix d_csrval8$o csr_mixed_complex
R 11783 5 415 csrmatrix d_csrrowptr csr_mixed_complex
R 11784 5 416 csrmatrix d_csrrowptr$sd csr_mixed_complex
R 11785 5 417 csrmatrix d_csrrowptr$p csr_mixed_complex
R 11786 5 418 csrmatrix d_csrrowptr$o csr_mixed_complex
R 11789 5 421 csrmatrix d_csrcolidx csr_mixed_complex
R 11790 5 422 csrmatrix d_csrcolidx$sd csr_mixed_complex
R 11791 5 423 csrmatrix d_csrcolidx$p csr_mixed_complex
R 11792 5 424 csrmatrix d_csrcolidx$o csr_mixed_complex
R 11794 5 426 csrmatrix info csr_mixed_complex
R 11795 5 427 csrmatrix infolu csr_mixed_complex
R 11796 5 428 csrmatrix descr csr_mixed_complex
R 11797 5 429 csrmatrix descrlu csr_mixed_complex
R 11798 5 430 csrmatrix nr csr_mixed_complex
R 11799 5 431 csrmatrix nc csr_mixed_complex
R 11800 5 432 csrmatrix nnz csr_mixed_complex
R 11801 5 433 csrmatrix nnzomp csr_mixed_complex
R 11802 5 434 csrmatrix nthread csr_mixed_complex
R 11803 5 435 csrmatrix lalloc csr_mixed_complex
R 11804 5 436 csrmatrix lfinalized csr_mixed_complex
R 11805 5 437 csrmatrix ldevice csr_mixed_complex
R 11806 5 438 csrmatrix lparallel csr_mixed_complex
R 11864 26 496 csrmatrix =
R 11892 26 524 csrmatrix -
S 12326 23 5 0 0 0 12329 624 50364 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 full2csr
S 12327 7 3 1 0 7470 1 12326 5727 20000004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 12328 1 3 3 0 7164 1 12326 50373 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a_csr
S 12329 14 5 0 0 0 1 12326 50364 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 7257 2 0 0 0 0 0 0 0 0 0 0 0 0 7 0 624 0 0 0 0 full2csr
F 12329 2 12327 12328
S 12330 6 1 0 0 6 1 12326 46684 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_1
S 12331 6 1 0 0 6 1 12326 46700 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_1
S 12332 6 1 0 0 6 1 12326 46708 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_1
S 12333 6 1 0 0 6 1 12326 46724 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_5_1
S 12334 6 1 0 0 6 1 12326 46732 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_1
S 12335 6 1 0 0 6 1 12326 50379 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2908
S 12336 6 1 0 0 6 1 12326 50388 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2911
S 12337 23 5 0 0 0 12340 624 50397 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 full2csrz
S 12338 7 3 1 0 7473 1 12337 5727 20000004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 12339 1 3 3 0 7233 1 12337 50373 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a_csr
S 12340 14 5 0 0 0 1 12337 50397 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 7260 2 0 0 0 0 0 0 0 0 0 0 0 0 39 0 624 0 0 0 0 full2csrz
F 12340 2 12338 12339
S 12341 6 1 0 0 6 1 12337 46684 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_1
S 12342 6 1 0 0 6 1 12337 46700 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_1
S 12343 6 1 0 0 6 1 12337 46708 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_1
S 12344 6 1 0 0 6 1 12337 46724 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_5_1
S 12345 6 1 0 0 6 1 12337 46732 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_1
S 12346 6 1 0 0 6 1 12337 50407 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2915
S 12347 6 1 0 0 6 1 12337 50416 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2918
S 12348 23 5 0 0 0 12351 624 50425 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 fullz2csrz
S 12349 7 3 1 0 7476 1 12348 5727 20000004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 12350 1 3 3 0 7233 1 12348 50373 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a_csr
S 12351 14 5 0 0 0 1 12348 50425 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 7263 2 0 0 0 0 0 0 0 0 0 0 0 0 72 0 624 0 0 0 0 fullz2csrz
F 12351 2 12349 12350
S 12352 6 1 0 0 6 1 12348 46684 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_1
S 12353 6 1 0 0 6 1 12348 46700 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_1
S 12354 6 1 0 0 6 1 12348 46708 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_1
S 12355 6 1 0 0 6 1 12348 46724 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_5_1
S 12356 6 1 0 0 6 1 12348 46732 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_1
S 12357 6 1 0 0 6 1 12348 50436 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2922
S 12358 6 1 0 0 6 1 12348 50445 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2925
S 12359 23 5 0 0 0 12362 624 50454 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 csr2full
S 12360 7 3 3 0 7479 1 12359 5727 20000004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 12361 1 3 1 0 7164 1 12359 50373 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a_csr
S 12362 14 5 0 0 0 1 12359 50454 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 7266 2 0 0 0 0 0 0 0 0 0 0 0 0 105 0 624 0 0 0 0 csr2full
F 12362 2 12360 12361
S 12363 6 1 0 0 6 1 12359 46684 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_1
S 12364 6 1 0 0 6 1 12359 46700 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_1
S 12365 6 1 0 0 6 1 12359 46708 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_1
S 12366 6 1 0 0 6 1 12359 46724 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_5_1
S 12367 6 1 0 0 6 1 12359 46732 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_1
S 12368 6 1 0 0 6 1 12359 50463 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2929
S 12369 6 1 0 0 6 1 12359 50472 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2932
S 12370 23 5 0 0 0 12373 624 50481 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 csrz2full
S 12371 7 3 3 0 7482 1 12370 5727 20000004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 12372 1 3 1 0 7233 1 12370 50373 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a_csr
S 12373 14 5 0 0 0 1 12370 50481 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 7269 2 0 0 0 0 0 0 0 0 0 0 0 0 134 0 624 0 0 0 0 csrz2full
F 12373 2 12371 12372
S 12374 6 1 0 0 6 1 12370 46684 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_1
S 12375 6 1 0 0 6 1 12370 46700 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_1
S 12376 6 1 0 0 6 1 12370 46708 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_1
S 12377 6 1 0 0 6 1 12370 46724 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_5_1
S 12378 6 1 0 0 6 1 12370 46732 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_1
S 12379 6 1 0 0 6 1 12370 50491 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2936
S 12380 6 1 0 0 6 1 12370 50500 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2939
S 12381 23 5 0 0 0 12384 624 50509 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 csr2fullz
S 12382 7 3 3 0 7485 1 12381 5727 20000004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 12383 1 3 1 0 7164 1 12381 50373 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a_csr
S 12384 14 5 0 0 0 1 12381 50509 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 7272 2 0 0 0 0 0 0 0 0 0 0 0 0 163 0 624 0 0 0 0 csr2fullz
F 12384 2 12382 12383
S 12385 6 1 0 0 6 1 12381 46684 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_1
S 12386 6 1 0 0 6 1 12381 46700 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_1
S 12387 6 1 0 0 6 1 12381 46708 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_1
S 12388 6 1 0 0 6 1 12381 46724 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_5_1
S 12389 6 1 0 0 6 1 12381 46732 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_1
S 12390 6 1 0 0 6 1 12381 50519 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2943
S 12391 6 1 0 0 6 1 12381 50528 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2946
S 12392 23 5 0 0 0 12395 624 50537 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 csrz2fullz
S 12393 7 3 3 0 7488 1 12392 5727 20000004 10003000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a
S 12394 1 3 1 0 7233 1 12392 50373 4 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a_csr
S 12395 14 5 0 0 0 1 12392 50537 20000000 400000 A 0 0 0 0 B 0 0 0 0 0 0 0 7275 2 0 0 0 0 0 0 0 0 0 0 0 0 192 0 624 0 0 0 0 csrz2fullz
F 12395 2 12393 12394
S 12396 6 1 0 0 6 1 12392 46684 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_0_1
S 12397 6 1 0 0 6 1 12392 46700 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_2_1
S 12398 6 1 0 0 6 1 12392 46708 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_3_1
S 12399 6 1 0 0 6 1 12392 46724 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_5_1
S 12400 6 1 0 0 6 1 12392 46732 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_b_6_1
S 12401 6 1 0 0 6 1 12392 50548 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2950
S 12402 6 1 0 0 6 1 12392 50557 40800006 3000 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 z_e_2953
A 67 1 0 0 0 56 679 0 0 0 0 0 0 0 0 0 0 0 0 0
A 70 1 0 0 0 62 681 0 0 0 0 0 0 0 0 0 0 0 0 0
A 86 1 0 0 0 92 719 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2899 2 0 0 2088 16 11368 0 0 0 2899 0 0 0 0 0 0 0 0 0
A 2900 1 0 0 1903 6 12334 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2901 1 0 0 2353 6 12330 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2902 1 0 0 2019 6 12335 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2903 1 0 0 2643 6 12332 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2904 1 0 0 2779 6 12331 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2905 1 0 0 2804 6 12336 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2906 1 0 0 2697 6 12333 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2907 1 0 0 2129 6 12345 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2908 1 0 0 2705 6 12341 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2909 1 0 0 2381 6 12346 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2910 1 0 0 2354 6 12343 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2911 1 0 0 2234 6 12342 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2912 1 0 0 2795 6 12347 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2913 1 0 0 2428 6 12344 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2914 1 0 0 2024 6 12356 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2915 1 0 0 1439 6 12352 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2916 1 0 0 2721 6 12357 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2917 1 0 0 1797 6 12354 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2918 1 0 0 1796 6 12353 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2919 1 0 0 2357 6 12358 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2920 1 0 0 2803 6 12355 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2921 1 0 0 2132 6 12367 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2922 1 0 0 2811 6 12363 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2923 1 0 0 2133 6 12368 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2924 1 0 0 2729 6 12365 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2925 1 0 0 2239 6 12364 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2926 1 0 0 2524 6 12369 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2927 1 0 0 2131 6 12366 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2928 1 0 0 2742 6 12378 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2929 1 0 0 2569 6 12374 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2930 1 0 0 2827 6 12379 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2931 1 0 0 2376 6 12376 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2932 1 0 0 1143 6 12375 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2933 1 0 0 2032 6 12380 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2934 1 0 0 2029 6 12377 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2935 1 0 0 2833 6 12389 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2936 1 0 0 2896 6 12385 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2937 1 0 0 2464 6 12390 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2938 1 0 0 2840 6 12387 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2939 1 0 0 2750 6 12386 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2940 1 0 0 2755 6 12391 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2941 1 0 0 1146 6 12388 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2942 1 0 0 2853 6 12400 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2943 1 0 0 2535 6 12396 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2944 1 0 0 2037 6 12401 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2945 1 0 0 2034 6 12398 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2946 1 0 0 2516 6 12397 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2947 1 0 0 2543 6 12402 0 0 0 0 0 0 0 0 0 0 0 0 0
A 2948 1 0 0 2763 6 12399 0 0 0 0 0 0 0 0 0 0 0 0 0
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
T 11369 7026 0 3 0 0
A 11436 16 0 0 1 2899 1
A 11437 16 0 0 1 2899 1
A 11438 16 0 0 1 2899 1
A 11439 16 0 0 1 2899 0
T 11440 7095 0 3 0 0
A 11507 16 0 0 1 2899 1
A 11508 16 0 0 1 2899 1
A 11509 16 0 0 1 2899 1
A 11510 16 0 0 1 2899 0
T 11511 7164 0 3 0 0
A 11578 16 0 0 1 2899 1
A 11579 16 0 0 1 2899 1
A 11580 16 0 0 1 2899 1
A 11581 16 0 0 1 2899 0
T 11582 7233 0 3 0 0
A 11649 16 0 0 1 2899 1
A 11650 16 0 0 1 2899 1
A 11651 16 0 0 1 2899 1
A 11652 16 0 0 1 2899 0
T 11653 7302 0 3 0 0
A 11726 16 0 0 1 2899 1
A 11727 16 0 0 1 2899 1
A 11728 16 0 0 1 2899 1
A 11729 16 0 0 1 2899 0
T 11730 7377 0 3 0 0
A 11803 16 0 0 1 2899 1
A 11804 16 0 0 1 2899 1
A 11805 16 0 0 1 2899 1
A 11806 16 0 0 1 2899 0
Z
