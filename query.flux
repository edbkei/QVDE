from(bucket: "sensors")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "fall_detection_accelerometer")
  |> limit(n: 10)
