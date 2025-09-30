# TNK-AXIN2 Experiment for Mub

Experiment ID: Mubs TNK experiment

Experiment Date: worked on this 9/29 and 9/30

Channels used and order: 
FLAG-plasmid 647 [0]
TNK-ab 594 [1]
GFP-condensate [2]
dapi [3]

Cell segmentation tool: cellposeSAM

Thresholding for puncta detection:
1. took away oversaturated cells in both plasmid-flag and tnk antibody channels
2. hresholded for GFP+ cells by 500 (based on imageJ and histogram of intensity)
3. thresholded for FLAG+ cells 2000 (based on imageJ and histogram of intensity)
4. created masks based on local intensity + STD*2 and dilated by a factor of 1 - checked cell masks for accuracy by looking over proofs
5. thresholding condensate area by 500 based on looking at distribution of sizes before threshold 
