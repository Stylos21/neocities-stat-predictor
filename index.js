const tf = require('@tensorflow/tfjs');
const neocities = require('./neocities.json');

const xs = tf.tensor2d(neocities.map(neocities => [
    neocities.day, neocities.month, neocities.year
]))

const ys = tf.tensor2d(neocities.map(color => [
    neocities.hits == "0-50" ? 1 : 0,
    neocities.hits == "50-100" ? 1 : 0,
    neocities.hits == "100-150" ? 1 : 0,
    neocities.hits == "150-200" ? 1 : 0,
    neocities.hits == "200-250" ? 1 : 0,
    neocities.hits == "250-300" ? 1 : 0,
    neocities.hits == "300-400" ? 1 : 0,
    neocities.hits == "400-500" ? 1 : 0,
    neocities.hits == "500-600" ? 1 : 0,
    neocities.hits == "600+" ? 1 : 0,
]))

//array with hit ranges (variable is named hitRanges)
const hitRanges = [
    "0-50",
    "50-100",
    "100-150",
    "150-200",
    "200-250",
    "250-300",
    "300-400",
    "400-500",
    "500-600",
    "600+"
]


const model = tf.sequential();
model.add(tf.layers.dense({
    inputShape: [3],
    activation: "softmax",
    units: 5
}))
model.add(tf.layers.dense({
    inputShape: [5],
    activation: "softmax",
    units: 10

}))


model.compile({
    loss: "categoricalCrossentropy",
    optimizer: tf.train.adam(0.1)
})
const d = new Date();
const month = d.getMonth() + 1;
const date = d.getDate();
const year = d.getFullYear();
// console.log(new Date().getMonth() + 1);
model.fit(xs, ys, { epochs: 1000 })
    .then(() => {
        const res = model.predict(tf.tensor2d([date, month, year], [1, 3]));
        const index = res.argMax(1).dataSync();
        console.log(hitRanges[index]);
    })