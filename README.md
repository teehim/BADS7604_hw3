# Bitcoin Price Prediction (POWER DS TEAM)

## Highlights
- จากผลการ Test Model ทั้งหมด พบว่า RNN/LSTM/GRU มีความแม่นยำในการทำนายที่สูงกว่า ARIMA มาก
- LSTM ใช้เวลาในการ Train น้อยที่สุด แต่ความแม่นยำต่ำที่สุดในกลุ่ม RNN ด้วยกัน 
- ในขณะที่ GRU ใช้เวลาในการ Train นานที่สุด แต่ความแม่นยำก็สูงที่สุดเช่นกัน

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
โดยที่มีเป้าหมายคือการทำนายราคาของ Bitcoin ของ 30 วันล่าสุด
เราจึงเลือกแบ่ง Data ตามจำนวนวัน

Train Data: 1110 วัน

Validation Data: 278 วัน

Test Data: 30 วัน

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

- ในส่วนของ Model ของเราจะเป็น Simple RNN 1 Layer โดยมี architecture ตามด้านล่าง

<img src="https://github.com/teehim/BADS7604_hw3/blob/main/images/rnn_arch.JPG?raw=true" style="width:500px;"/>

### LSTM
- Long Short Term Memory Network (LSTM) พัฒนาต่อมาจาก RNN ซึ่งทำงานได้ดีในการเรียนรู้แบบ Long-Term หลักการทำงานของ LSTM คือจะมี Weight กำหนดการลืม (Forget) ไว้ด้วย
  
  <a href="https://imgur.com/UDeu9ko"><img src="https://i.imgur.com/UDeu9ko.png" title="source: imgur.com" /></a>
- ในส่วนแรกจะมี Sigmoid Layer ซึ่งให้ค่าออกมาระหว่าง 0 กับ 1 จะได้ค่าออกมา ซึ่งจะนำไปใช้ในการคูนกับ State ก่อนในภายหลัง ซึ่งเป็นการปรับการใช้ State เก่า
  
  <a href="https://imgur.com/sCLFuC3"><img src="https://i.imgur.com/sCLFuC3.png" title="source: imgur.com" /></a>
  
- ในส่วนถัดไปเราก็จะมีการคำนวน (ในส่วนของ tanh) แล้วนำค่านั้นไปคูณกับค่าที่ได้จาก Sigmoid Layer เพื่อตั้งค่า Weight ในข้อมูลใหม่

  <a href="https://imgur.com/POmNkXo"><img src="https://i.imgur.com/POmNkXo.png" title="source: imgur.com" /></a>
  
- จากนั้นเราก็นำสองส่วนมารวมกัน ให้ลืมค่าเก่าบางส่วน และรับบางส่วนจากของใหม่มา จะได้ค่า Cell State

  <a href="https://imgur.com/xS4iw06"><img src="https://i.imgur.com/xS4iw06.png" title="source: imgur.com" /></a>
  
- จากนั้นเราก็นำค่า Cell State มาคำนวน (tanh) และนำค่าที่ได้มาคูณกับค่าจาก Sigmoid Layer เพื่อตั้ง Weight ให้อีกครั้ง และจะได้ออกมาเป็นค่า h(t)

   <a href="https://imgur.com/UPafMvS"><img src="https://i.imgur.com/UPafMvS.png" title="source: imgur.com" /></a>

- ในส่วนของ Model ของเราจะเป็น Stacked LSTM Layer โดยมี architecture ตามด้านล่าง

<img src="https://github.com/teehim/BADS7604_hw3/blob/main/images/lstm_arch.JPG?raw=true" style="width:500px;"/>

### GRU
- Gated Recurrent Units (GRU) เป็นกลไลปิดเปิดการอัพเดทสถานะภายใน Recurrent Neural Network ที่คล้ายกับ Long Short-Term Memory (LSTM) ที่จะมี Forget Gate แต่มี Parameter น้อยกว่า LSTM เนื่องจากไม่มี Output Gate
- GRU มีประสิทธิภาพใกล้เคียงกับ LSTM ในหลาย ๆ งาน แต่เนื่องจาก Parameter น้อยกว่าทำให้เทรนได้ง่ายกว่า เร็วกว่า และในบางงานที่ DataSet มีขนาดเล็ก พบว่า GRU ประสิทธิภาพดีกว่า

  <a href="https://imgur.com/n6MHlE4"><img src="https://i.imgur.com/n6MHlE4.png" title="source: imgur.com" /></a>

- ในส่วนของ Model ของเราจะเป็น Stacked GRU Layer โดยมี architecture ตามด้านล่าง

<img src="https://github.com/teehim/BADS7604_hw3/blob/main/images/gru_arch.JPG?raw=true" style="width:500px;"/>

## Training

#### ARIMA

    เวลาที่ใช้ในการ Train 88.233 วินาที

#### RNN

    Trained on 1 NVIDIA GeForce RTX 3080

    - Optimizer: Adam
    - Learning rate: 0.001
    - Loss Function: Mean Squeare Error
    - Batch size: 128
    - Epoch: 300

    เวลาที่ใช้ในการ Train เฉลี่ย 21.47 ms/epoch

#### LSTM

    Trained on 1 NVIDIA GeForce RTX 3080

    - Optimizer: Adam
    - Learning rate: 0.001
    - Loss Function: Mean Squeare Error
    - Batch size: 128
    - Epoch: 300

    เวลาที่ใช้ในการ Train เฉลี่ย 10.84 ms/epoch

#### GRU

    Trained on 1 NVIDIA GeForce RTX 3080

    - Optimizer: Adam
    - Learning rate: 0.001
    - Loss Function: Mean Squeare Error
    - Batch size: 128
    - Epoch: 300

    เวลาที่ใช้ในการ Train เฉลี่ย 33.04 ms/epoch

## Results
จากการ Train, Test และ Validation ได้ค่า Mean Square Error ออกมาดังนี้
MSE
- ARIMA: 9634641032.254011
- RNN: 2196504059.13288
- LSTM: 2621728661.542714
- GRU: 2014365225.5934842

## Discussion
จากค่า Mean Square Error ที่ได้ออกมา ทำให้สรุปได้ว่า ARIMA Model มีความแม่นยำในการนายราคา Bitcoin ต่ำที่สุด

ส่วนโมเดล Neural Network ทั้ง 3 ที่ทีมใช้มีค่า Mean Square Error ที่ใกล้เคียงกัน แต่น้อยกว่า ARIMA Model ค่อนข้างมากโดย
  - LSTM มีค่า Mean Square Error สูงที่สุดในกลุ่ม Neural Network ทั้งหมด
  - RNN มีค่า Mean Square Error น้อยเป็นอันดับที่ 2
  - ส่วน GRU นั้น มีค่า Mean Square Error ที่น้อยที่สุด

จึงสรุปได้ว่า Neural Network นั้น มีประสิทธิภาพในการทำนายข้อมูลประเภท Time Series ที่สูงกว่า ARIMA Model ค่อนข้างมาก และเมื่อเปรียบเทียบในกลุ่ม Neural Network ด้วยกัน พบว่า ความแม่นยำนั้น เรียงลำดับได้ดังนี้

  1.Gated Recurrent Units (GRU)
  
  2.Recurrent Neural Network (RNN)
  
  3.Long Short Term Memory Network (LSTM)

## Conclusion
- การใช้ Neural Network ในการทำนายข้อมูล Bitcoin มีความแม่นยำกว่า ARIMA Model โมเดลมาก อาจจะประยุกต์ไปใช้กับการทำทายสินทรัพย์อื่นๆได้ โดยหากจะนำไปพัฒนาต่อ อาจจะเปลี่ยนช่วงข้อมูลที่มีความถี่มากขึ้น เช่น รายชั่วโมงหรือรายนาที และอาจเพิ่มการใช้ Technical Indicator เข้าไปเพิ่มในการเทรนโมเดล อาจจะให้ความแม่นยำที่มากกว่าการใช้เพียงราคาอย่างเดียว

## References
- https://www.facebook.com/investicbkk
- https://th.tradingview.com/
- https://github.com/StreamAlpha/tvdatafeed
- https://sirawich99.medium.com/%E0%B8%AA%E0%B8%A3%E0%B8%B8%E0%B8%9B%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B9%80%E0%B8%82%E0%B9%89%E0%B8%B2%E0%B9%83%E0%B8%88-rnn-lstm-gru-24-10-2020-95602afe3053

## Members
- (20%) 6220422048 กชกร เรืองศรี (Train RNN Model)
- (0%)  6220422061 ไตรเทพ จันทร์เทพ (ติดต่อไม่ได้)
- (20%) 6220422065 สุธาสินี โพธิ์แจ่ม (Train LSTM Model)
- (20%) 6310422028 วรเมธ ปลอดโปร่ง (ดึงข้อมูล, ช่วย Train LSTM Model, สรุปผล )
- (20%) 6310422031 ธนัตถ์กรณ์ ชื่นบรรลือสุข (Train GRU Model)
- (20%) 6310422046 วีระศักดิ์ การุณย์ (Train ARIMA Model)

#### งานชึ้นนี้เป็นส่วนหนึ่งของวิชา BADS7604 การเรียนรู้เชิงลึก (Deep Learning) คณะสถิติประยุกต์ หลักสูตรวิทยาศาสตรมหาบัณฑิต การวิเคราะห์ธุรกิจและวิทยาการข้อมูล สถาบันบัณฑิตพัฒนบริหารศาสตร์
