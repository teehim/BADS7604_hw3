# Bitcoin Price Prediction (POWER DS TEAM)

## Highlights


## Introduction
กลุ่มของเราอยากทราบว่าโมเดล Deep Learning กลุ่ม RNN/LSTM/GRU นั้นสามารถที่จะทำนายราคาของ Bitcoin ได้หรือไม่ โดยกลุ่มของเราได้ทำการเทียบกับโมเดลทางสถิติที่ใช้ในการทำนายข้อมูลที่เป็น Time Series อย่าง ARIMA (AutoRegressive Integrated Moving Average)

## Data
<a href="https://imgur.com/JxzRWgE"><img src="https://i.imgur.com/JxzRWgE.png" title="source: imgur.com" /></a>

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
### ARIMA Model
- ARIMA model หรือ ชื่อเต็มๆ ก็คือ “(AutoRegressive Integrated Moving Average)” ประกอบไปด้วย 3 เทคนิค คือ AR (Autoregressive), I (Integrated), MA (Moving average) โดยทุกเทคนิค คือการร่วมกันกำจัด “Noise” ออกจากข้อมูลเพื่อพยายามลด Error term ให้ได้มากสุดจนสามารถมั่นใจได้ว่า ข้อมูลนั้น Reliable หรือ เชื่อถือได้ ซึ่งจะทำให้การทำนาย (Forecast) มีประสิทธิภาพมากขึ้น
- กระบวนการเช็คข้อมูลก่อนจะทำ ARIMA 
  - 1.1 — ตรวจดูว่าข้อมูล Time Series ของเรา มี “White noise” ที่ไม่สามารถกำจัดมันออกไปได้หรือไม่? ถ้ามี แปลว่า Time Series นั้นๆ มัน Uncorrelated กันครับ เช่น ราคาในแต่ละวันนั้นแรนด้อมจากกันโดยสิ้นเชิง 
  - 1.2 — ข้อมูลนเป็น Stationary หรือไม่ ถ้าไม่ ก็ Differencing ให้เป็น Stationary ก่อน จึงเอา ARIMA มาใช้งานกับข้อมูลนั้นๆ ได้
-องค์ประกอบของ ARIMA(p,d,q) ประกอบด้วยพารามิเตอร์ 3 ตัว จาก 3 วิธีการที่นำมารวมเป็น โมเดล ARIMA โดยที่ p เป็นพารามิเตอร์ของ AR, d เป็นพารามิเตอร์ของ I, q เป็นพารามิเตอร์ของ MA
  - Integrated (I)
    - พารามิเตอร์ d ของวิธีการ Integrated ตัวนี้นำมาหาความเป็น Stationary ของ Time Series ซึ่งต้องคงสถานะนี้เมื่อเวลาผ่านไป เป็นสิ่งที่ สำคัญมากในโมเดล ARIMA ก็คือ ข้อมูลอย่าง ค่าเฉลี่ย (Mean), ความแปรปรวน (Variance) ต้องมีค่าคงที่เมื่อเวลาผ่านไปกล่าวคือ ถ้าข้อมูลมีแนวโน้ม (Trend) หรือ มี Seasonal มันก็จะไม่ใช่ Stationary แล้ว เพราะข้อมูลจำพวก Mean และ Variance จะไม่คงที่ เราจึงต้องทำให้ข้อมูลเป็น Stationary ก่อนจึงจะสร้างโมเดลจากข้อมูลนั้นๆ ได้
  - Auto Regressive (AR)
    - Auto Regressive ตามความหมายก็คือ การใช้ค่าของตัวมันเองในการทำนายวันถัดไป โดยที่พารามิเตอร์ p ของ AR ก็คือ จำนวนเวลาที่ lag ใน time series โดยการคำนวนจะมีการถ่วงน้ำหนักด้วยข้อมูล time series ก่อนหน้าหน้า p อีกทีหนึง สรุปคือ สมการนี้ตั้งอยู่บนสมมุติฐานที่ว่า ค่าของวันที่เราจะทำนายมีความสัมพันธ์กับค่าในอดีตของตัวมันเอง
  - Moving Average (MA)
    - คอนเซปของ MA นี้นั้นจะค่อนข้างคล้ายกับคอนเซปของ AR แต่จะตั้งบนสมมุติฐานของ Error แทน เช่น ถ้าพารามิเตอร์ q ของ MA เป็น 2 เป้าหมาย ณ วันนี้จะเท่ากับ Error ของเมื่อวาน * coefficient + ด้วย Error ของเมื่อวานซืน * coefficient

### RNN
- Recurrent Neural Network (RNN) คือ Artificial Neural Network แบบหนึ่งที่ออกแบบมาแก้ปัญหาสำหรับงานที่ข้อมูลมีลำดับ Sequence โดยใช้หลักการ Feed สถานะภายในของโมเดล กลับมาเป็น Input ใหม่ คู่กับ Input ปกติ เรียกว่า Hidden State, Internal State, Memory ช่วยให้โมเดลรู้จำ Pattern ของลำดับ Input Sequence ได้ 

<a href="https://imgur.com/wR8hPG8"><img src="https://i.imgur.com/wR8hPG8.png" title="source: imgur.com" /></a>
- ซึ่งก็จะทำให้ออกมาเหมือนกับ Neural Network ธรรมดา ที่มีหลายๆ ตัว และต่อ Output เข้า Network ตัวใหม่ โดยสมการของ Recurrent Neural Network คือ

<a href="https://imgur.com/smqLoEZ"><img src="https://i.imgur.com/smqLoEZ.png" title="source: imgur.com" /></a>
- โดยสมการนี้แสดงถึงการที่ใช้ค่า Output ของ x(t) ร่วมกับ Output ของ h(t-1) (หรือ Output ของ Network ที่แล้ว) โดยมี Weight 2 ตัวปรับของ x(t) กับ h(t-1)
- เนื่องจาก RNN ใช้ข้อมูลจาก Network ก่อนๆ ทำให้สามารถทำงานได้ดีในข้อมูลแบบ Time Series (นำข้อมูลเวลาก่อนๆ มาหาต่อกับเวลาปัจจุบัน) ซึ่ง Time Series นั้นรวมถึงข้อมูลแบบ Text และข้อมูลเสียง
- ในทางทฤษดีนั้น RNN จะสามารถแก้ปัญหาแบบ Long Term ได้ดีมากๆ ถ้าเลือก Weight ได้ดี แต่ในการใช้จริงนั้นอาจจะไม่ได้ออกมาดึอย่างที่เราคาดไว้ เช่น “ผมเป็นคนไทย ผมสามารถพูดภาษา _(ไทย)_” ซึ่งคำว่า “ผม” “พูด” “ภาษา” นั้นไม่ได้ช่วยในการตอบคำถามมากนัก แต่ในการเลือก Weight ออกมาจริงๆ นั้น มันทำให้ข้อมูลสำคัญบางส่วนหายไปด้วย ซึ่ง LSTM (Long Short Term Memory) Network สามารถแก้ปัญหานี้ได้

### LSTM

### GRU

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
- https://www.facebook.com/investicbkk
- https://th.tradingview.com/
- https://github.com/StreamAlpha/tvdatafeed
- https://sirawich99.medium.com/%E0%B8%AA%E0%B8%A3%E0%B8%B8%E0%B8%9B%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B9%80%E0%B8%82%E0%B9%89%E0%B8%B2%E0%B9%83%E0%B8%88-rnn-lstm-gru-24-10-2020-95602afe3053
