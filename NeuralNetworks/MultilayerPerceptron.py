"""
a multilayer perceptron learning to recognize handwritten digits

the network starts with a bunch of neurons corresponding to each of the 28x28 pixels in an input image digit
i.e 28x28 = 784 neurons in total

each of these neurons holds a number that represent the gray scale value of the corresponding pixel, ranging from
0 for black pixels up to 1 for white pixels

the number inside each neuron is called the 'activation' implying that the neuron is lit up when it's activated i.e
when the activation is a high number

these 784 neurons make up the 1st layer of our network

the last layer has 10 neurons each representing one of the digits 0123456789

the activation in each of these neurons (again some number between 0 and 1) represents how much the system thinks that
a given image corresponds with a given digit

the hidden layers (in between 1st and last layers) is currently a black box - how on earth should we handle this
process of recognizing digits?

activations in one layer always determine the activations in the next

the brightest neuron in the output layer is the network's choice for what digit this image represents

each connection (from neuron in one layer to neuron in another) is weighted. for example from the 1st layer, weighted
sum of activations:

w1a1 + w2a2 + ... + wnan -- pump this into a fxn f(x) that squishes it to a value between 0 and 1. f(x) can be the
sigmoid func sig(x) =       1
                        ----------
                         1 + e^(-x)

so the activation of a neuron on the 2nd layer (by neurons on the previous layer) is a measure of how positive the
relevant weighted sum sig( 1a1 + w2a2 + ... + wnan) is

if you want the neuron to light up when the weighted sum > a number other than 0 i.e you want some bias for it to
be inactive, add in a negative number for the (bias for inactivity) term like so:
sig(1a1 + w2a2 + ... + wnan + (bias for inactivity))

so the weights tell me what pixel pattern this neuron on the 2nd layer is picking up on and the bias tells how high
the weighted sum needs to be before the neuron gets active

now picture this for every single neuron in that 2nd layer i.e being connected to all 784 pixel neurons from
the previous layer. each of the 784 connections has its own weight associated with it and some bias that you added
to the weighted sum before squishing it with the sigmoid func

sig(input vector  x    matrix    +  bias vector)
 _  _     _                 _            _  _
| a0 |   | w0,0 w0,1 ... w0n |          | b0 |
| a1 |   | w1,0 w1,1 ... w1n |          | b1 |
| .  |   | .     .        .  |   +      | .  |            ==> sig(Wa + b)
| .  |   | .     .        .  |          | .  |
| .  |   | .     .        .  |          | .  |
| an |   | wk,0 wk,1 ... wk,n|          | bn |
-    -    -                  -          -    -

learning a multilayered perceptron therefore entails finding a valid set of weights and biases that solves the
problem at hand

it helps so solve issues (like when the network doesn't perform as anticipated or if it works but not for the
reasons I expected) by asking myself what are my weights and biases doing?

the above process is termed as forward propagation then back propagation is that learned info (on the magnitude of
change to reduce the error) being transferred backwards through our network, to adjust the weights based on this info
so it's gonna be an iterative cycle of forward and back propagation (on multiple inputs) till the weights are assigned
such that our network predicts correctly. then that brings our training process to the end!

oh and btw neural networks are super inspired by the brain!

and basically deep learning is a subset of machine learning that's super inspired by the brain's neural networks
"""