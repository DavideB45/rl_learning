const int sensorPin = A0; 

void setup() {
  Serial.begin(115200); 
}

void loop() {
  // Only execute if the Mac has sent something
  if (Serial.available() > 0) {
    
    // Read the incoming byte to clear it from the buffer
    char incomingByte = Serial.read(); 
    
    // Take a fresh reading and send it back immediately
    int rawADC = analogRead(sensorPin);
    Serial.println(rawADC);
  }
}