mkdir -p tmp
wget https://archive.ics.uci.edu/static/public/222/bank+marketing.zip -P ./tmp/
unzip ./tmp/bank+marketing.zip -d ./tmp/bank+marketing
unzip ./tmp/bank+marketing/bank.zip -d ./tmp/bank
cp ./tmp/bank/bank-full.csv ./
rm -rf ./tmp