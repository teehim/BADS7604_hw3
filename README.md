# Bitcoin Price Prediction (POWER DS TEAM)

## Highlights


## Introduction
กลุ่มของเราอยากทราบว่าโมเดล Deep Learning กลุ่ม RNN/LSTM/GRU นั้นสามารถที่จะทำนายราคาของ Bitcoin ได้หรือไม่ โดยกลุ่มของเราได้ทำการเทียบกับโมเดลทางสถิติที่ใช้ในการทำนายข้อมูลที่เป็น Time Series อย่าง ARIMA (AutoRegressive Integrated Moving Average)

## Data


#### Data source
กลุ่มเราได้ทำการดึงข้อมูลราคา Bitcoin มาจาก www.tradingview.com ผ่าน Library TvDatafeed
โดย กลุ่มของเราดึงราคา BTC คู่กับ THB จาก BITKUB โดยข้อมูลมีช่วงเวลาเป็นรายวัน ราคาเปิด/ปิด ราคาสูงสุด/ต่ำสุด และปริมาณการซื้อขายในแต่ละช่วง

#### Data pre-processing
กลุ่มของเราใช้ Min-Max Normalize ในการ Pre-processing ของข้อมูล โดยกลุ่มเราใช้ราคาปิดในการเทรนโมเดล

#### Data splitting
เราได้ทำการแบ่ง data ออกเป็น 3 ส่วนดังนี้

train ...%

validation ...%

test ...%

## Network architecture
ARIMA Model

RNN

LSTM

GRU

## Training



## Results

MSE
- ARIMA: 9634641032.254011
- RNN: 2196502267.229943
- LSTM: 2052927766.0934634
- GRU: 2014365225.5934842

## Discussion


## Conclusion


## References
