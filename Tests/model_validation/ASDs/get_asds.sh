wget https://dcc.ligo.org/public/0165/T2000012/002/AplusDesign.txt
wget https://dcc.ligo.org/public/0165/T2000012/002/avirgo_O5high_NEW.txt
wget https://apps.et-gw.eu/tds/?call_file=18213_ET10kmcolumns.txt --output-document=ET10kmcolumns.txt
wget https://apps.et-gw.eu/tds/?call_file=18213_ET15kmcolumns.txt --output-document=ET15kmcolumns.txt
wget https://apps.et-gw.eu/tds/?call_file=18213_ET20kmcolumns.txt --output-document=ET20kmcolumns.txt
python extract_ET_ASDs.py
