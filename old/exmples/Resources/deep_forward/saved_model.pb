ۧ

??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
layer_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer_1/kernel
y
"layer_1/kernel/Read/ReadVariableOpReadVariableOplayer_1/kernel*&
_output_shapes
:*
dtype0
p
layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer_1/bias
i
 layer_1/bias/Read/ReadVariableOpReadVariableOplayer_1/bias*
_output_shapes
:*
dtype0
?
layer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer_3/kernel
y
"layer_3/kernel/Read/ReadVariableOpReadVariableOplayer_3/kernel*&
_output_shapes
:*
dtype0
p
layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer_3/bias
i
 layer_3/bias/Read/ReadVariableOpReadVariableOplayer_3/bias*
_output_shapes
:*
dtype0
z
layer_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namelayer_7/kernel
s
"layer_7/kernel/Read/ReadVariableOpReadVariableOplayer_7/kernel* 
_output_shapes
:
??*
dtype0
q
layer_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer_7/bias
j
 layer_7/bias/Read/ReadVariableOpReadVariableOplayer_7/bias*
_output_shapes	
:?*
dtype0
y
layer_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*
shared_namelayer_8/kernel
r
"layer_8/kernel/Read/ReadVariableOpReadVariableOplayer_8/kernel*
_output_shapes
:	?2*
dtype0
p
layer_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namelayer_8/bias
i
 layer_8/bias/Read/ReadVariableOpReadVariableOplayer_8/bias*
_output_shapes
:2*
dtype0
x
layer_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*
shared_namelayer_9/kernel
q
"layer_9/kernel/Read/ReadVariableOpReadVariableOplayer_9/kernel*
_output_shapes

:2
*
dtype0
p
layer_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namelayer_9/bias
i
 layer_9/bias/Read/ReadVariableOpReadVariableOplayer_9/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/layer_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/layer_1/kernel/m
?
)Adam/layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_1/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/layer_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/layer_1/bias/m
w
'Adam/layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/layer_3/kernel/m
?
)Adam/layer_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_3/kernel/m*&
_output_shapes
:*
dtype0
~
Adam/layer_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/layer_3/bias/m
w
'Adam/layer_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/layer_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/layer_7/kernel/m
?
)Adam/layer_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_7/kernel/m* 
_output_shapes
:
??*
dtype0

Adam/layer_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/layer_7/bias/m
x
'Adam/layer_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_7/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/layer_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*&
shared_nameAdam/layer_8/kernel/m
?
)Adam/layer_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_8/kernel/m*
_output_shapes
:	?2*
dtype0
~
Adam/layer_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/layer_8/bias/m
w
'Adam/layer_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_8/bias/m*
_output_shapes
:2*
dtype0
?
Adam/layer_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*&
shared_nameAdam/layer_9/kernel/m

)Adam/layer_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer_9/kernel/m*
_output_shapes

:2
*
dtype0
~
Adam/layer_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/layer_9/bias/m
w
'Adam/layer_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer_9/bias/m*
_output_shapes
:
*
dtype0
?
Adam/layer_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/layer_1/kernel/v
?
)Adam/layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_1/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/layer_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/layer_1/bias/v
w
'Adam/layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/layer_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/layer_3/kernel/v
?
)Adam/layer_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_3/kernel/v*&
_output_shapes
:*
dtype0
~
Adam/layer_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/layer_3/bias/v
w
'Adam/layer_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/layer_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*&
shared_nameAdam/layer_7/kernel/v
?
)Adam/layer_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_7/kernel/v* 
_output_shapes
:
??*
dtype0

Adam/layer_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/layer_7/bias/v
x
'Adam/layer_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_7/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/layer_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*&
shared_nameAdam/layer_8/kernel/v
?
)Adam/layer_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_8/kernel/v*
_output_shapes
:	?2*
dtype0
~
Adam/layer_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*$
shared_nameAdam/layer_8/bias/v
w
'Adam/layer_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_8/bias/v*
_output_shapes
:2*
dtype0
?
Adam/layer_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2
*&
shared_nameAdam/layer_9/kernel/v

)Adam/layer_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer_9/kernel/v*
_output_shapes

:2
*
dtype0
~
Adam/layer_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/layer_9/bias/v
w
'Adam/layer_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer_9/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
 trainable_variables
!	variables
"regularization_losses
#	keras_api
R
$trainable_variables
%	variables
&regularization_losses
'	keras_api
R
(trainable_variables
)	variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
h

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
?
>iter

?beta_1

@beta_2
	Adecay
Blearning_ratem?m?m?m?,m?-m?2m?3m?8m?9m?v?v?v?v?,v?-v?2v?3v?8v?9v?
F
0
1
2
3
,4
-5
26
37
88
99
 
F
0
1
2
3
,4
-5
26
37
88
99
?
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
regularization_losses
Fnon_trainable_variables
trainable_variables

Glayers
 
ZX
VARIABLE_VALUElayer_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
trainable_variables
	variables
regularization_losses
Knon_trainable_variables

Llayers
 
 
 
?
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
trainable_variables
	variables
regularization_losses
Pnon_trainable_variables

Qlayers
ZX
VARIABLE_VALUElayer_3/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_3/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
trainable_variables
	variables
regularization_losses
Unon_trainable_variables

Vlayers
 
 
 
?
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
 trainable_variables
!	variables
"regularization_losses
Znon_trainable_variables

[layers
 
 
 
?
\metrics
]layer_regularization_losses
^layer_metrics
$trainable_variables
%	variables
&regularization_losses
_non_trainable_variables

`layers
 
 
 
?
ametrics
blayer_regularization_losses
clayer_metrics
(trainable_variables
)	variables
*regularization_losses
dnon_trainable_variables

elayers
ZX
VARIABLE_VALUElayer_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
?
fmetrics
glayer_regularization_losses
hlayer_metrics
.trainable_variables
/	variables
0regularization_losses
inon_trainable_variables

jlayers
ZX
VARIABLE_VALUElayer_8/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_8/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
 
?
kmetrics
llayer_regularization_losses
mlayer_metrics
4trainable_variables
5	variables
6regularization_losses
nnon_trainable_variables

olayers
ZX
VARIABLE_VALUElayer_9/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUElayer_9/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
?
pmetrics
qlayer_regularization_losses
rlayer_metrics
:trainable_variables
;	variables
<regularization_losses
snon_trainable_variables

tlayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

u0
v1
 
 
 
?
0
1
2
3
4
5
6
7
	8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	wtotal
	xcount
y	variables
z	keras_api
D
	{total
	|count
}
_fn_kwargs
~	variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

w0
x1

y	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1

~	variables
}{
VARIABLE_VALUEAdam/layer_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_3/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_3/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_7/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_7/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_8/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_8/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_9/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_9/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_3/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_3/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_7/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_7/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_8/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_8/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/layer_9/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/layer_9/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_layer_1_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_layer_1_inputlayer_1/kernellayer_1/biaslayer_3/kernellayer_3/biaslayer_7/kernellayer_7/biaslayer_8/kernellayer_8/biaslayer_9/kernellayer_9/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_9431
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"layer_1/kernel/Read/ReadVariableOp layer_1/bias/Read/ReadVariableOp"layer_3/kernel/Read/ReadVariableOp layer_3/bias/Read/ReadVariableOp"layer_7/kernel/Read/ReadVariableOp layer_7/bias/Read/ReadVariableOp"layer_8/kernel/Read/ReadVariableOp layer_8/bias/Read/ReadVariableOp"layer_9/kernel/Read/ReadVariableOp layer_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/layer_1/kernel/m/Read/ReadVariableOp'Adam/layer_1/bias/m/Read/ReadVariableOp)Adam/layer_3/kernel/m/Read/ReadVariableOp'Adam/layer_3/bias/m/Read/ReadVariableOp)Adam/layer_7/kernel/m/Read/ReadVariableOp'Adam/layer_7/bias/m/Read/ReadVariableOp)Adam/layer_8/kernel/m/Read/ReadVariableOp'Adam/layer_8/bias/m/Read/ReadVariableOp)Adam/layer_9/kernel/m/Read/ReadVariableOp'Adam/layer_9/bias/m/Read/ReadVariableOp)Adam/layer_1/kernel/v/Read/ReadVariableOp'Adam/layer_1/bias/v/Read/ReadVariableOp)Adam/layer_3/kernel/v/Read/ReadVariableOp'Adam/layer_3/bias/v/Read/ReadVariableOp)Adam/layer_7/kernel/v/Read/ReadVariableOp'Adam/layer_7/bias/v/Read/ReadVariableOp)Adam/layer_8/kernel/v/Read/ReadVariableOp'Adam/layer_8/bias/v/Read/ReadVariableOp)Adam/layer_9/kernel/v/Read/ReadVariableOp'Adam/layer_9/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_9854
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_1/kernellayer_1/biaslayer_3/kernellayer_3/biaslayer_7/kernellayer_7/biaslayer_8/kernellayer_8/biaslayer_9/kernellayer_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/layer_1/kernel/mAdam/layer_1/bias/mAdam/layer_3/kernel/mAdam/layer_3/bias/mAdam/layer_7/kernel/mAdam/layer_7/bias/mAdam/layer_8/kernel/mAdam/layer_8/bias/mAdam/layer_9/kernel/mAdam/layer_9/bias/mAdam/layer_1/kernel/vAdam/layer_1/bias/vAdam/layer_3/kernel/vAdam/layer_3/bias/vAdam/layer_7/kernel/vAdam/layer_7/bias/vAdam/layer_8/kernel/vAdam/layer_8/bias/vAdam/layer_9/kernel/vAdam/layer_9/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_9981??
?	
?
A__inference_layer_7_layer_call_and_return_conditional_losses_9665

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
)__inference_sequential_layer_call_fn_9396
layer_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_93732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_namelayer_1_input
?
{
&__inference_layer_7_layer_call_fn_9674

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_7_layer_call_and_return_conditional_losses_91752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
]
A__inference_layer_2_layer_call_and_return_conditional_losses_9042

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_layer_6_layer_call_and_return_conditional_losses_9649

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????w  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_9551

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_93152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_layer_9_layer_call_fn_9714

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_9_layer_call_and_return_conditional_losses_92292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
`
A__inference_layer_5_layer_call_and_return_conditional_losses_9628

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
&__inference_layer_5_layer_call_fn_9638

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_5_layer_call_and_return_conditional_losses_91322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_layer_1_layer_call_fn_9596

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_1_layer_call_and_return_conditional_losses_90752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_layer_3_layer_call_and_return_conditional_losses_9103

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_layer_1_layer_call_and_return_conditional_losses_9587

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
)__inference_sequential_layer_call_fn_9338
layer_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_93152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_namelayer_1_input
?
`
A__inference_layer_5_layer_call_and_return_conditional_losses_9132

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_layer_3_layer_call_fn_9616

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_3_layer_call_and_return_conditional_losses_91032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
A__inference_layer_5_layer_call_and_return_conditional_losses_9137

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?R
?
__inference__traced_save_9854
file_prefix-
)savev2_layer_1_kernel_read_readvariableop+
'savev2_layer_1_bias_read_readvariableop-
)savev2_layer_3_kernel_read_readvariableop+
'savev2_layer_3_bias_read_readvariableop-
)savev2_layer_7_kernel_read_readvariableop+
'savev2_layer_7_bias_read_readvariableop-
)savev2_layer_8_kernel_read_readvariableop+
'savev2_layer_8_bias_read_readvariableop-
)savev2_layer_9_kernel_read_readvariableop+
'savev2_layer_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_layer_1_kernel_m_read_readvariableop2
.savev2_adam_layer_1_bias_m_read_readvariableop4
0savev2_adam_layer_3_kernel_m_read_readvariableop2
.savev2_adam_layer_3_bias_m_read_readvariableop4
0savev2_adam_layer_7_kernel_m_read_readvariableop2
.savev2_adam_layer_7_bias_m_read_readvariableop4
0savev2_adam_layer_8_kernel_m_read_readvariableop2
.savev2_adam_layer_8_bias_m_read_readvariableop4
0savev2_adam_layer_9_kernel_m_read_readvariableop2
.savev2_adam_layer_9_bias_m_read_readvariableop4
0savev2_adam_layer_1_kernel_v_read_readvariableop2
.savev2_adam_layer_1_bias_v_read_readvariableop4
0savev2_adam_layer_3_kernel_v_read_readvariableop2
.savev2_adam_layer_3_bias_v_read_readvariableop4
0savev2_adam_layer_7_kernel_v_read_readvariableop2
.savev2_adam_layer_7_bias_v_read_readvariableop4
0savev2_adam_layer_8_kernel_v_read_readvariableop2
.savev2_adam_layer_8_bias_v_read_readvariableop4
0savev2_adam_layer_9_kernel_v_read_readvariableop2
.savev2_adam_layer_9_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_layer_1_kernel_read_readvariableop'savev2_layer_1_bias_read_readvariableop)savev2_layer_3_kernel_read_readvariableop'savev2_layer_3_bias_read_readvariableop)savev2_layer_7_kernel_read_readvariableop'savev2_layer_7_bias_read_readvariableop)savev2_layer_8_kernel_read_readvariableop'savev2_layer_8_bias_read_readvariableop)savev2_layer_9_kernel_read_readvariableop'savev2_layer_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_layer_1_kernel_m_read_readvariableop.savev2_adam_layer_1_bias_m_read_readvariableop0savev2_adam_layer_3_kernel_m_read_readvariableop.savev2_adam_layer_3_bias_m_read_readvariableop0savev2_adam_layer_7_kernel_m_read_readvariableop.savev2_adam_layer_7_bias_m_read_readvariableop0savev2_adam_layer_8_kernel_m_read_readvariableop.savev2_adam_layer_8_bias_m_read_readvariableop0savev2_adam_layer_9_kernel_m_read_readvariableop.savev2_adam_layer_9_bias_m_read_readvariableop0savev2_adam_layer_1_kernel_v_read_readvariableop.savev2_adam_layer_1_bias_v_read_readvariableop0savev2_adam_layer_3_kernel_v_read_readvariableop.savev2_adam_layer_3_bias_v_read_readvariableop0savev2_adam_layer_7_kernel_v_read_readvariableop.savev2_adam_layer_7_bias_v_read_readvariableop0savev2_adam_layer_8_kernel_v_read_readvariableop.savev2_adam_layer_8_bias_v_read_readvariableop0savev2_adam_layer_9_kernel_v_read_readvariableop.savev2_adam_layer_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::
??:?:	?2:2:2
:
: : : : : : : : : :::::
??:?:	?2:2:2
:
:::::
??:?:	?2:2:2
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?2: 

_output_shapes
:2:$	 

_output_shapes

:2
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?2: 

_output_shapes
:2:$ 

_output_shapes

:2
: 

_output_shapes
:
:,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::&""
 
_output_shapes
:
??:!#

_output_shapes	
:?:%$!

_output_shapes
:	?2: %

_output_shapes
:2:$& 

_output_shapes

:2
: '

_output_shapes
:
:(

_output_shapes
: 
?C
?
__inference__wrapped_model_9036
layer_1_input5
1sequential_layer_1_conv2d_readvariableop_resource6
2sequential_layer_1_biasadd_readvariableop_resource5
1sequential_layer_3_conv2d_readvariableop_resource6
2sequential_layer_3_biasadd_readvariableop_resource5
1sequential_layer_7_matmul_readvariableop_resource6
2sequential_layer_7_biasadd_readvariableop_resource5
1sequential_layer_8_matmul_readvariableop_resource6
2sequential_layer_8_biasadd_readvariableop_resource5
1sequential_layer_9_matmul_readvariableop_resource6
2sequential_layer_9_biasadd_readvariableop_resource
identity??)sequential/layer_1/BiasAdd/ReadVariableOp?(sequential/layer_1/Conv2D/ReadVariableOp?)sequential/layer_3/BiasAdd/ReadVariableOp?(sequential/layer_3/Conv2D/ReadVariableOp?)sequential/layer_7/BiasAdd/ReadVariableOp?(sequential/layer_7/MatMul/ReadVariableOp?)sequential/layer_8/BiasAdd/ReadVariableOp?(sequential/layer_8/MatMul/ReadVariableOp?)sequential/layer_9/BiasAdd/ReadVariableOp?(sequential/layer_9/MatMul/ReadVariableOp?
(sequential/layer_1/Conv2D/ReadVariableOpReadVariableOp1sequential_layer_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(sequential/layer_1/Conv2D/ReadVariableOp?
sequential/layer_1/Conv2DConv2Dlayer_1_input0sequential/layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
sequential/layer_1/Conv2D?
)sequential/layer_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/layer_1/BiasAdd/ReadVariableOp?
sequential/layer_1/BiasAddBiasAdd"sequential/layer_1/Conv2D:output:01sequential/layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential/layer_1/BiasAdd?
sequential/layer_1/ReluRelu#sequential/layer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
sequential/layer_1/Relu?
sequential/layer_2/MaxPoolMaxPool%sequential/layer_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
sequential/layer_2/MaxPool?
(sequential/layer_3/Conv2D/ReadVariableOpReadVariableOp1sequential_layer_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(sequential/layer_3/Conv2D/ReadVariableOp?
sequential/layer_3/Conv2DConv2D#sequential/layer_2/MaxPool:output:00sequential/layer_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
2
sequential/layer_3/Conv2D?
)sequential/layer_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/layer_3/BiasAdd/ReadVariableOp?
sequential/layer_3/BiasAddBiasAdd"sequential/layer_3/Conv2D:output:01sequential/layer_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2
sequential/layer_3/BiasAdd?
sequential/layer_3/ReluRelu#sequential/layer_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
sequential/layer_3/Relu?
sequential/layer_4/MaxPoolMaxPool%sequential/layer_3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
sequential/layer_4/MaxPool?
sequential/layer_5/IdentityIdentity#sequential/layer_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
sequential/layer_5/Identity?
sequential/layer_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????w  2
sequential/layer_6/Const?
sequential/layer_6/ReshapeReshape$sequential/layer_5/Identity:output:0!sequential/layer_6/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/layer_6/Reshape?
(sequential/layer_7/MatMul/ReadVariableOpReadVariableOp1sequential_layer_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(sequential/layer_7/MatMul/ReadVariableOp?
sequential/layer_7/MatMulMatMul#sequential/layer_6/Reshape:output:00sequential/layer_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/layer_7/MatMul?
)sequential/layer_7/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)sequential/layer_7/BiasAdd/ReadVariableOp?
sequential/layer_7/BiasAddBiasAdd#sequential/layer_7/MatMul:product:01sequential/layer_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential/layer_7/BiasAdd?
sequential/layer_7/ReluRelu#sequential/layer_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential/layer_7/Relu?
(sequential/layer_8/MatMul/ReadVariableOpReadVariableOp1sequential_layer_8_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02*
(sequential/layer_8/MatMul/ReadVariableOp?
sequential/layer_8/MatMulMatMul%sequential/layer_7/Relu:activations:00sequential/layer_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer_8/MatMul?
)sequential/layer_8/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_8_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02+
)sequential/layer_8/BiasAdd/ReadVariableOp?
sequential/layer_8/BiasAddBiasAdd#sequential/layer_8/MatMul:product:01sequential/layer_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/layer_8/BiasAdd?
sequential/layer_8/ReluRelu#sequential/layer_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential/layer_8/Relu?
(sequential/layer_9/MatMul/ReadVariableOpReadVariableOp1sequential_layer_9_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02*
(sequential/layer_9/MatMul/ReadVariableOp?
sequential/layer_9/MatMulMatMul%sequential/layer_8/Relu:activations:00sequential/layer_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/layer_9/MatMul?
)sequential/layer_9/BiasAdd/ReadVariableOpReadVariableOp2sequential_layer_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/layer_9/BiasAdd/ReadVariableOp?
sequential/layer_9/BiasAddBiasAdd#sequential/layer_9/MatMul:product:01sequential/layer_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/layer_9/BiasAdd?
sequential/layer_9/SoftmaxSoftmax#sequential/layer_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential/layer_9/Softmax?
IdentityIdentity$sequential/layer_9/Softmax:softmax:0*^sequential/layer_1/BiasAdd/ReadVariableOp)^sequential/layer_1/Conv2D/ReadVariableOp*^sequential/layer_3/BiasAdd/ReadVariableOp)^sequential/layer_3/Conv2D/ReadVariableOp*^sequential/layer_7/BiasAdd/ReadVariableOp)^sequential/layer_7/MatMul/ReadVariableOp*^sequential/layer_8/BiasAdd/ReadVariableOp)^sequential/layer_8/MatMul/ReadVariableOp*^sequential/layer_9/BiasAdd/ReadVariableOp)^sequential/layer_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2V
)sequential/layer_1/BiasAdd/ReadVariableOp)sequential/layer_1/BiasAdd/ReadVariableOp2T
(sequential/layer_1/Conv2D/ReadVariableOp(sequential/layer_1/Conv2D/ReadVariableOp2V
)sequential/layer_3/BiasAdd/ReadVariableOp)sequential/layer_3/BiasAdd/ReadVariableOp2T
(sequential/layer_3/Conv2D/ReadVariableOp(sequential/layer_3/Conv2D/ReadVariableOp2V
)sequential/layer_7/BiasAdd/ReadVariableOp)sequential/layer_7/BiasAdd/ReadVariableOp2T
(sequential/layer_7/MatMul/ReadVariableOp(sequential/layer_7/MatMul/ReadVariableOp2V
)sequential/layer_8/BiasAdd/ReadVariableOp)sequential/layer_8/BiasAdd/ReadVariableOp2T
(sequential/layer_8/MatMul/ReadVariableOp(sequential/layer_8/MatMul/ReadVariableOp2V
)sequential/layer_9/BiasAdd/ReadVariableOp)sequential/layer_9/BiasAdd/ReadVariableOp2T
(sequential/layer_9/MatMul/ReadVariableOp(sequential/layer_9/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:?????????
'
_user_specified_namelayer_1_input
?
]
A__inference_layer_6_layer_call_and_return_conditional_losses_9156

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????w  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
A__inference_layer_5_layer_call_and_return_conditional_losses_9633

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_layer_2_layer_call_fn_9048

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_2_layer_call_and_return_conditional_losses_90422
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_9981
file_prefix#
assignvariableop_layer_1_kernel#
assignvariableop_1_layer_1_bias%
!assignvariableop_2_layer_3_kernel#
assignvariableop_3_layer_3_bias%
!assignvariableop_4_layer_7_kernel#
assignvariableop_5_layer_7_bias%
!assignvariableop_6_layer_8_kernel#
assignvariableop_7_layer_8_bias%
!assignvariableop_8_layer_9_kernel#
assignvariableop_9_layer_9_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1-
)assignvariableop_19_adam_layer_1_kernel_m+
'assignvariableop_20_adam_layer_1_bias_m-
)assignvariableop_21_adam_layer_3_kernel_m+
'assignvariableop_22_adam_layer_3_bias_m-
)assignvariableop_23_adam_layer_7_kernel_m+
'assignvariableop_24_adam_layer_7_bias_m-
)assignvariableop_25_adam_layer_8_kernel_m+
'assignvariableop_26_adam_layer_8_bias_m-
)assignvariableop_27_adam_layer_9_kernel_m+
'assignvariableop_28_adam_layer_9_bias_m-
)assignvariableop_29_adam_layer_1_kernel_v+
'assignvariableop_30_adam_layer_1_bias_v-
)assignvariableop_31_adam_layer_3_kernel_v+
'assignvariableop_32_adam_layer_3_bias_v-
)assignvariableop_33_adam_layer_7_kernel_v+
'assignvariableop_34_adam_layer_7_bias_v-
)assignvariableop_35_adam_layer_8_kernel_v+
'assignvariableop_36_adam_layer_8_bias_v-
)assignvariableop_37_adam_layer_9_kernel_v+
'assignvariableop_38_adam_layer_9_bias_v
identity_40??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_layer_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_layer_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_layer_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_layer_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_layer_7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_layer_7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_layer_8_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_layer_8_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_layer_9_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_layer_9_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_layer_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_layer_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_layer_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_layer_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_layer_7_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_layer_7_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_layer_8_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_layer_8_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_layer_9_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_layer_9_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_layer_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_layer_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_layer_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_layer_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_layer_7_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_layer_7_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_layer_8_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_layer_8_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_layer_9_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_layer_9_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39?
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?&
?
D__inference_sequential_layer_call_and_return_conditional_losses_9373

inputs
layer_1_9343
layer_1_9345
layer_3_9349
layer_3_9351
layer_7_9357
layer_7_9359
layer_8_9362
layer_8_9364
layer_9_9367
layer_9_9369
identity??layer_1/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?layer_7/StatefulPartitionedCall?layer_8/StatefulPartitionedCall?layer_9/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCallinputslayer_1_9343layer_1_9345*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_1_layer_call_and_return_conditional_losses_90752!
layer_1/StatefulPartitionedCall?
layer_2/PartitionedCallPartitionedCall(layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_2_layer_call_and_return_conditional_losses_90422
layer_2/PartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall layer_2/PartitionedCall:output:0layer_3_9349layer_3_9351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_3_layer_call_and_return_conditional_losses_91032!
layer_3/StatefulPartitionedCall?
layer_4/PartitionedCallPartitionedCall(layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_4_layer_call_and_return_conditional_losses_90542
layer_4/PartitionedCall?
layer_5/PartitionedCallPartitionedCall layer_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_5_layer_call_and_return_conditional_losses_91372
layer_5/PartitionedCall?
layer_6/PartitionedCallPartitionedCall layer_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_6_layer_call_and_return_conditional_losses_91562
layer_6/PartitionedCall?
layer_7/StatefulPartitionedCallStatefulPartitionedCall layer_6/PartitionedCall:output:0layer_7_9357layer_7_9359*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_7_layer_call_and_return_conditional_losses_91752!
layer_7/StatefulPartitionedCall?
layer_8/StatefulPartitionedCallStatefulPartitionedCall(layer_7/StatefulPartitionedCall:output:0layer_8_9362layer_8_9364*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_8_layer_call_and_return_conditional_losses_92022!
layer_8/StatefulPartitionedCall?
layer_9/StatefulPartitionedCallStatefulPartitionedCall(layer_8/StatefulPartitionedCall:output:0layer_9_9367layer_9_9369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_9_layer_call_and_return_conditional_losses_92292!
layer_9/StatefulPartitionedCall?
IdentityIdentity(layer_9/StatefulPartitionedCall:output:0 ^layer_1/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall ^layer_7/StatefulPartitionedCall ^layer_8/StatefulPartitionedCall ^layer_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall2B
layer_7/StatefulPartitionedCalllayer_7/StatefulPartitionedCall2B
layer_8/StatefulPartitionedCalllayer_8/StatefulPartitionedCall2B
layer_9/StatefulPartitionedCalllayer_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_layer_6_layer_call_fn_9654

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_6_layer_call_and_return_conditional_losses_91562
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_layer_8_layer_call_and_return_conditional_losses_9685

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
D__inference_sequential_layer_call_and_return_conditional_losses_9482

inputs*
&layer_1_conv2d_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource*
&layer_3_conv2d_readvariableop_resource+
'layer_3_biasadd_readvariableop_resource*
&layer_7_matmul_readvariableop_resource+
'layer_7_biasadd_readvariableop_resource*
&layer_8_matmul_readvariableop_resource+
'layer_8_biasadd_readvariableop_resource*
&layer_9_matmul_readvariableop_resource+
'layer_9_biasadd_readvariableop_resource
identity??layer_1/BiasAdd/ReadVariableOp?layer_1/Conv2D/ReadVariableOp?layer_3/BiasAdd/ReadVariableOp?layer_3/Conv2D/ReadVariableOp?layer_7/BiasAdd/ReadVariableOp?layer_7/MatMul/ReadVariableOp?layer_8/BiasAdd/ReadVariableOp?layer_8/MatMul/ReadVariableOp?layer_9/BiasAdd/ReadVariableOp?layer_9/MatMul/ReadVariableOp?
layer_1/Conv2D/ReadVariableOpReadVariableOp&layer_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer_1/Conv2D/ReadVariableOp?
layer_1/Conv2DConv2Dinputs%layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
layer_1/Conv2D?
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layer_1/BiasAdd/ReadVariableOp?
layer_1/BiasAddBiasAddlayer_1/Conv2D:output:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
layer_1/BiasAddx
layer_1/ReluRelulayer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
layer_1/Relu?
layer_2/MaxPoolMaxPoollayer_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
layer_2/MaxPool?
layer_3/Conv2D/ReadVariableOpReadVariableOp&layer_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer_3/Conv2D/ReadVariableOp?
layer_3/Conv2DConv2Dlayer_2/MaxPool:output:0%layer_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
2
layer_3/Conv2D?
layer_3/BiasAdd/ReadVariableOpReadVariableOp'layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layer_3/BiasAdd/ReadVariableOp?
layer_3/BiasAddBiasAddlayer_3/Conv2D:output:0&layer_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2
layer_3/BiasAddx
layer_3/ReluRelulayer_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
layer_3/Relu?
layer_4/MaxPoolMaxPoollayer_3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
layer_4/MaxPools
layer_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_5/dropout/Const?
layer_5/dropout/MulMullayer_4/MaxPool:output:0layer_5/dropout/Const:output:0*
T0*/
_output_shapes
:?????????2
layer_5/dropout/Mulv
layer_5/dropout/ShapeShapelayer_4/MaxPool:output:0*
T0*
_output_shapes
:2
layer_5/dropout/Shape?
,layer_5/dropout/random_uniform/RandomUniformRandomUniformlayer_5/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????*
dtype02.
,layer_5/dropout/random_uniform/RandomUniform?
layer_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
layer_5/dropout/GreaterEqual/y?
layer_5/dropout/GreaterEqualGreaterEqual5layer_5/dropout/random_uniform/RandomUniform:output:0'layer_5/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????2
layer_5/dropout/GreaterEqual?
layer_5/dropout/CastCast layer_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????2
layer_5/dropout/Cast?
layer_5/dropout/Mul_1Mullayer_5/dropout/Mul:z:0layer_5/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????2
layer_5/dropout/Mul_1o
layer_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????w  2
layer_6/Const?
layer_6/ReshapeReshapelayer_5/dropout/Mul_1:z:0layer_6/Const:output:0*
T0*(
_output_shapes
:??????????2
layer_6/Reshape?
layer_7/MatMul/ReadVariableOpReadVariableOp&layer_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer_7/MatMul/ReadVariableOp?
layer_7/MatMulMatMullayer_6/Reshape:output:0%layer_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_7/MatMul?
layer_7/BiasAdd/ReadVariableOpReadVariableOp'layer_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_7/BiasAdd/ReadVariableOp?
layer_7/BiasAddBiasAddlayer_7/MatMul:product:0&layer_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_7/BiasAddq
layer_7/ReluRelulayer_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer_7/Relu?
layer_8/MatMul/ReadVariableOpReadVariableOp&layer_8_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
layer_8/MatMul/ReadVariableOp?
layer_8/MatMulMatMullayer_7/Relu:activations:0%layer_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer_8/MatMul?
layer_8/BiasAdd/ReadVariableOpReadVariableOp'layer_8_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer_8/BiasAdd/ReadVariableOp?
layer_8/BiasAddBiasAddlayer_8/MatMul:product:0&layer_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer_8/BiasAddp
layer_8/ReluRelulayer_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
layer_8/Relu?
layer_9/MatMul/ReadVariableOpReadVariableOp&layer_9_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
layer_9/MatMul/ReadVariableOp?
layer_9/MatMulMatMullayer_8/Relu:activations:0%layer_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
layer_9/MatMul?
layer_9/BiasAdd/ReadVariableOpReadVariableOp'layer_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
layer_9/BiasAdd/ReadVariableOp?
layer_9/BiasAddBiasAddlayer_9/MatMul:product:0&layer_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
layer_9/BiasAddy
layer_9/SoftmaxSoftmaxlayer_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
layer_9/Softmax?
IdentityIdentitylayer_9/Softmax:softmax:0^layer_1/BiasAdd/ReadVariableOp^layer_1/Conv2D/ReadVariableOp^layer_3/BiasAdd/ReadVariableOp^layer_3/Conv2D/ReadVariableOp^layer_7/BiasAdd/ReadVariableOp^layer_7/MatMul/ReadVariableOp^layer_8/BiasAdd/ReadVariableOp^layer_8/MatMul/ReadVariableOp^layer_9/BiasAdd/ReadVariableOp^layer_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2>
layer_1/Conv2D/ReadVariableOplayer_1/Conv2D/ReadVariableOp2@
layer_3/BiasAdd/ReadVariableOplayer_3/BiasAdd/ReadVariableOp2>
layer_3/Conv2D/ReadVariableOplayer_3/Conv2D/ReadVariableOp2@
layer_7/BiasAdd/ReadVariableOplayer_7/BiasAdd/ReadVariableOp2>
layer_7/MatMul/ReadVariableOplayer_7/MatMul/ReadVariableOp2@
layer_8/BiasAdd/ReadVariableOplayer_8/BiasAdd/ReadVariableOp2>
layer_8/MatMul/ReadVariableOplayer_8/MatMul/ReadVariableOp2@
layer_9/BiasAdd/ReadVariableOplayer_9/BiasAdd/ReadVariableOp2>
layer_9/MatMul/ReadVariableOplayer_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_9431
layer_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_90362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_namelayer_1_input
?6
?
D__inference_sequential_layer_call_and_return_conditional_losses_9526

inputs*
&layer_1_conv2d_readvariableop_resource+
'layer_1_biasadd_readvariableop_resource*
&layer_3_conv2d_readvariableop_resource+
'layer_3_biasadd_readvariableop_resource*
&layer_7_matmul_readvariableop_resource+
'layer_7_biasadd_readvariableop_resource*
&layer_8_matmul_readvariableop_resource+
'layer_8_biasadd_readvariableop_resource*
&layer_9_matmul_readvariableop_resource+
'layer_9_biasadd_readvariableop_resource
identity??layer_1/BiasAdd/ReadVariableOp?layer_1/Conv2D/ReadVariableOp?layer_3/BiasAdd/ReadVariableOp?layer_3/Conv2D/ReadVariableOp?layer_7/BiasAdd/ReadVariableOp?layer_7/MatMul/ReadVariableOp?layer_8/BiasAdd/ReadVariableOp?layer_8/MatMul/ReadVariableOp?layer_9/BiasAdd/ReadVariableOp?layer_9/MatMul/ReadVariableOp?
layer_1/Conv2D/ReadVariableOpReadVariableOp&layer_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer_1/Conv2D/ReadVariableOp?
layer_1/Conv2DConv2Dinputs%layer_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
layer_1/Conv2D?
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layer_1/BiasAdd/ReadVariableOp?
layer_1/BiasAddBiasAddlayer_1/Conv2D:output:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
layer_1/BiasAddx
layer_1/ReluRelulayer_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
layer_1/Relu?
layer_2/MaxPoolMaxPoollayer_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
layer_2/MaxPool?
layer_3/Conv2D/ReadVariableOpReadVariableOp&layer_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
layer_3/Conv2D/ReadVariableOp?
layer_3/Conv2DConv2Dlayer_2/MaxPool:output:0%layer_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
2
layer_3/Conv2D?
layer_3/BiasAdd/ReadVariableOpReadVariableOp'layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layer_3/BiasAdd/ReadVariableOp?
layer_3/BiasAddBiasAddlayer_3/Conv2D:output:0&layer_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2
layer_3/BiasAddx
layer_3/ReluRelulayer_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
layer_3/Relu?
layer_4/MaxPoolMaxPoollayer_3/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
layer_4/MaxPool?
layer_5/IdentityIdentitylayer_4/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
layer_5/Identityo
layer_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"????w  2
layer_6/Const?
layer_6/ReshapeReshapelayer_5/Identity:output:0layer_6/Const:output:0*
T0*(
_output_shapes
:??????????2
layer_6/Reshape?
layer_7/MatMul/ReadVariableOpReadVariableOp&layer_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
layer_7/MatMul/ReadVariableOp?
layer_7/MatMulMatMullayer_6/Reshape:output:0%layer_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_7/MatMul?
layer_7/BiasAdd/ReadVariableOpReadVariableOp'layer_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_7/BiasAdd/ReadVariableOp?
layer_7/BiasAddBiasAddlayer_7/MatMul:product:0&layer_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
layer_7/BiasAddq
layer_7/ReluRelulayer_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
layer_7/Relu?
layer_8/MatMul/ReadVariableOpReadVariableOp&layer_8_matmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
layer_8/MatMul/ReadVariableOp?
layer_8/MatMulMatMullayer_7/Relu:activations:0%layer_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer_8/MatMul?
layer_8/BiasAdd/ReadVariableOpReadVariableOp'layer_8_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
layer_8/BiasAdd/ReadVariableOp?
layer_8/BiasAddBiasAddlayer_8/MatMul:product:0&layer_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
layer_8/BiasAddp
layer_8/ReluRelulayer_8/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
layer_8/Relu?
layer_9/MatMul/ReadVariableOpReadVariableOp&layer_9_matmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
layer_9/MatMul/ReadVariableOp?
layer_9/MatMulMatMullayer_8/Relu:activations:0%layer_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
layer_9/MatMul?
layer_9/BiasAdd/ReadVariableOpReadVariableOp'layer_9_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
layer_9/BiasAdd/ReadVariableOp?
layer_9/BiasAddBiasAddlayer_9/MatMul:product:0&layer_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
layer_9/BiasAddy
layer_9/SoftmaxSoftmaxlayer_9/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
layer_9/Softmax?
IdentityIdentitylayer_9/Softmax:softmax:0^layer_1/BiasAdd/ReadVariableOp^layer_1/Conv2D/ReadVariableOp^layer_3/BiasAdd/ReadVariableOp^layer_3/Conv2D/ReadVariableOp^layer_7/BiasAdd/ReadVariableOp^layer_7/MatMul/ReadVariableOp^layer_8/BiasAdd/ReadVariableOp^layer_8/MatMul/ReadVariableOp^layer_9/BiasAdd/ReadVariableOp^layer_9/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2>
layer_1/Conv2D/ReadVariableOplayer_1/Conv2D/ReadVariableOp2@
layer_3/BiasAdd/ReadVariableOplayer_3/BiasAdd/ReadVariableOp2>
layer_3/Conv2D/ReadVariableOplayer_3/Conv2D/ReadVariableOp2@
layer_7/BiasAdd/ReadVariableOplayer_7/BiasAdd/ReadVariableOp2>
layer_7/MatMul/ReadVariableOplayer_7/MatMul/ReadVariableOp2@
layer_8/BiasAdd/ReadVariableOplayer_8/BiasAdd/ReadVariableOp2>
layer_8/MatMul/ReadVariableOplayer_8/MatMul/ReadVariableOp2@
layer_9/BiasAdd/ReadVariableOplayer_9/BiasAdd/ReadVariableOp2>
layer_9/MatMul/ReadVariableOplayer_9/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
D__inference_sequential_layer_call_and_return_conditional_losses_9246
layer_1_input
layer_1_9086
layer_1_9088
layer_3_9114
layer_3_9116
layer_7_9186
layer_7_9188
layer_8_9213
layer_8_9215
layer_9_9240
layer_9_9242
identity??layer_1/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?layer_5/StatefulPartitionedCall?layer_7/StatefulPartitionedCall?layer_8/StatefulPartitionedCall?layer_9/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCalllayer_1_inputlayer_1_9086layer_1_9088*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_1_layer_call_and_return_conditional_losses_90752!
layer_1/StatefulPartitionedCall?
layer_2/PartitionedCallPartitionedCall(layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_2_layer_call_and_return_conditional_losses_90422
layer_2/PartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall layer_2/PartitionedCall:output:0layer_3_9114layer_3_9116*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_3_layer_call_and_return_conditional_losses_91032!
layer_3/StatefulPartitionedCall?
layer_4/PartitionedCallPartitionedCall(layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_4_layer_call_and_return_conditional_losses_90542
layer_4/PartitionedCall?
layer_5/StatefulPartitionedCallStatefulPartitionedCall layer_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_5_layer_call_and_return_conditional_losses_91322!
layer_5/StatefulPartitionedCall?
layer_6/PartitionedCallPartitionedCall(layer_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_6_layer_call_and_return_conditional_losses_91562
layer_6/PartitionedCall?
layer_7/StatefulPartitionedCallStatefulPartitionedCall layer_6/PartitionedCall:output:0layer_7_9186layer_7_9188*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_7_layer_call_and_return_conditional_losses_91752!
layer_7/StatefulPartitionedCall?
layer_8/StatefulPartitionedCallStatefulPartitionedCall(layer_7/StatefulPartitionedCall:output:0layer_8_9213layer_8_9215*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_8_layer_call_and_return_conditional_losses_92022!
layer_8/StatefulPartitionedCall?
layer_9/StatefulPartitionedCallStatefulPartitionedCall(layer_8/StatefulPartitionedCall:output:0layer_9_9240layer_9_9242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_9_layer_call_and_return_conditional_losses_92292!
layer_9/StatefulPartitionedCall?
IdentityIdentity(layer_9/StatefulPartitionedCall:output:0 ^layer_1/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall ^layer_5/StatefulPartitionedCall ^layer_7/StatefulPartitionedCall ^layer_8/StatefulPartitionedCall ^layer_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall2B
layer_5/StatefulPartitionedCalllayer_5/StatefulPartitionedCall2B
layer_7/StatefulPartitionedCalllayer_7/StatefulPartitionedCall2B
layer_8/StatefulPartitionedCalllayer_8/StatefulPartitionedCall2B
layer_9/StatefulPartitionedCalllayer_9/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_namelayer_1_input
?

?
A__inference_layer_3_layer_call_and_return_conditional_losses_9607

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
D__inference_sequential_layer_call_and_return_conditional_losses_9315

inputs
layer_1_9285
layer_1_9287
layer_3_9291
layer_3_9293
layer_7_9299
layer_7_9301
layer_8_9304
layer_8_9306
layer_9_9309
layer_9_9311
identity??layer_1/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?layer_5/StatefulPartitionedCall?layer_7/StatefulPartitionedCall?layer_8/StatefulPartitionedCall?layer_9/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCallinputslayer_1_9285layer_1_9287*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_1_layer_call_and_return_conditional_losses_90752!
layer_1/StatefulPartitionedCall?
layer_2/PartitionedCallPartitionedCall(layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_2_layer_call_and_return_conditional_losses_90422
layer_2/PartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall layer_2/PartitionedCall:output:0layer_3_9291layer_3_9293*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_3_layer_call_and_return_conditional_losses_91032!
layer_3/StatefulPartitionedCall?
layer_4/PartitionedCallPartitionedCall(layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_4_layer_call_and_return_conditional_losses_90542
layer_4/PartitionedCall?
layer_5/StatefulPartitionedCallStatefulPartitionedCall layer_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_5_layer_call_and_return_conditional_losses_91322!
layer_5/StatefulPartitionedCall?
layer_6/PartitionedCallPartitionedCall(layer_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_6_layer_call_and_return_conditional_losses_91562
layer_6/PartitionedCall?
layer_7/StatefulPartitionedCallStatefulPartitionedCall layer_6/PartitionedCall:output:0layer_7_9299layer_7_9301*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_7_layer_call_and_return_conditional_losses_91752!
layer_7/StatefulPartitionedCall?
layer_8/StatefulPartitionedCallStatefulPartitionedCall(layer_7/StatefulPartitionedCall:output:0layer_8_9304layer_8_9306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_8_layer_call_and_return_conditional_losses_92022!
layer_8/StatefulPartitionedCall?
layer_9/StatefulPartitionedCallStatefulPartitionedCall(layer_8/StatefulPartitionedCall:output:0layer_9_9309layer_9_9311*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_9_layer_call_and_return_conditional_losses_92292!
layer_9/StatefulPartitionedCall?
IdentityIdentity(layer_9/StatefulPartitionedCall:output:0 ^layer_1/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall ^layer_5/StatefulPartitionedCall ^layer_7/StatefulPartitionedCall ^layer_8/StatefulPartitionedCall ^layer_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall2B
layer_5/StatefulPartitionedCalllayer_5/StatefulPartitionedCall2B
layer_7/StatefulPartitionedCalllayer_7/StatefulPartitionedCall2B
layer_8/StatefulPartitionedCalllayer_8/StatefulPartitionedCall2B
layer_9/StatefulPartitionedCalllayer_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_layer_5_layer_call_fn_9643

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_5_layer_call_and_return_conditional_losses_91372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_layer_9_layer_call_and_return_conditional_losses_9705

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
]
A__inference_layer_4_layer_call_and_return_conditional_losses_9054

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
A__inference_layer_9_layer_call_and_return_conditional_losses_9229

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?

?
A__inference_layer_1_layer_call_and_return_conditional_losses_9075

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
D__inference_sequential_layer_call_and_return_conditional_losses_9279
layer_1_input
layer_1_9249
layer_1_9251
layer_3_9255
layer_3_9257
layer_7_9263
layer_7_9265
layer_8_9268
layer_8_9270
layer_9_9273
layer_9_9275
identity??layer_1/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?layer_7/StatefulPartitionedCall?layer_8/StatefulPartitionedCall?layer_9/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCalllayer_1_inputlayer_1_9249layer_1_9251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_1_layer_call_and_return_conditional_losses_90752!
layer_1/StatefulPartitionedCall?
layer_2/PartitionedCallPartitionedCall(layer_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_2_layer_call_and_return_conditional_losses_90422
layer_2/PartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall layer_2/PartitionedCall:output:0layer_3_9255layer_3_9257*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_3_layer_call_and_return_conditional_losses_91032!
layer_3/StatefulPartitionedCall?
layer_4/PartitionedCallPartitionedCall(layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_4_layer_call_and_return_conditional_losses_90542
layer_4/PartitionedCall?
layer_5/PartitionedCallPartitionedCall layer_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_5_layer_call_and_return_conditional_losses_91372
layer_5/PartitionedCall?
layer_6/PartitionedCallPartitionedCall layer_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_6_layer_call_and_return_conditional_losses_91562
layer_6/PartitionedCall?
layer_7/StatefulPartitionedCallStatefulPartitionedCall layer_6/PartitionedCall:output:0layer_7_9263layer_7_9265*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_7_layer_call_and_return_conditional_losses_91752!
layer_7/StatefulPartitionedCall?
layer_8/StatefulPartitionedCallStatefulPartitionedCall(layer_7/StatefulPartitionedCall:output:0layer_8_9268layer_8_9270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_8_layer_call_and_return_conditional_losses_92022!
layer_8/StatefulPartitionedCall?
layer_9/StatefulPartitionedCallStatefulPartitionedCall(layer_8/StatefulPartitionedCall:output:0layer_9_9273layer_9_9275*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_9_layer_call_and_return_conditional_losses_92292!
layer_9/StatefulPartitionedCall?
IdentityIdentity(layer_9/StatefulPartitionedCall:output:0 ^layer_1/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall ^layer_7/StatefulPartitionedCall ^layer_8/StatefulPartitionedCall ^layer_9/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall2B
layer_7/StatefulPartitionedCalllayer_7/StatefulPartitionedCall2B
layer_8/StatefulPartitionedCalllayer_8/StatefulPartitionedCall2B
layer_9/StatefulPartitionedCalllayer_9/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_namelayer_1_input
?
?
)__inference_sequential_layer_call_fn_9576

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_93732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
A__inference_layer_8_layer_call_and_return_conditional_losses_9202

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
{
&__inference_layer_8_layer_call_fn_9694

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_8_layer_call_and_return_conditional_losses_92022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
A__inference_layer_7_layer_call_and_return_conditional_losses_9175

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
B
&__inference_layer_4_layer_call_fn_9060

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_layer_4_layer_call_and_return_conditional_losses_90542
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
layer_1_input>
serving_default_layer_1_input:0?????????;
layer_90
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:ԝ
?F
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?B
_tf_keras_sequential?B{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_1_input"}}, {"class_name": "Conv2D", "config": {"name": "layer_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 15, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "layer_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "layer_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "layer_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "layer_7", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "layer_8", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "layer_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_1_input"}}, {"class_name": "Conv2D", "config": {"name": "layer_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 15, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "layer_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "layer_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "layer_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "layer_7", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "layer_8", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "layer_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 15, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 30]}}
?
 trainable_variables
!	variables
"regularization_losses
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "layer_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
$trainable_variables
%	variables
&regularization_losses
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "layer_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
(trainable_variables
)	variables
*regularization_losses
+	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "layer_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

,kernel
-bias
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_7", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 375}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 375]}}
?

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_8", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "layer_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_9", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
?
>iter

?beta_1

@beta_2
	Adecay
Blearning_ratem?m?m?m?,m?-m?2m?3m?8m?9m?v?v?v?v?,v?-v?2v?3v?8v?9v?"
	optimizer
f
0
1
2
3
,4
-5
26
37
88
99"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
,4
-5
26
37
88
99"
trackable_list_wrapper
?
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
regularization_losses
Fnon_trainable_variables
trainable_variables

Glayers
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
(:&2layer_1/kernel
:2layer_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
trainable_variables
	variables
regularization_losses
Knon_trainable_variables

Llayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
trainable_variables
	variables
regularization_losses
Pnon_trainable_variables

Qlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&2layer_3/kernel
:2layer_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
trainable_variables
	variables
regularization_losses
Unon_trainable_variables

Vlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
 trainable_variables
!	variables
"regularization_losses
Znon_trainable_variables

[layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
\metrics
]layer_regularization_losses
^layer_metrics
$trainable_variables
%	variables
&regularization_losses
_non_trainable_variables

`layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ametrics
blayer_regularization_losses
clayer_metrics
(trainable_variables
)	variables
*regularization_losses
dnon_trainable_variables

elayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2layer_7/kernel
:?2layer_7/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
fmetrics
glayer_regularization_losses
hlayer_metrics
.trainable_variables
/	variables
0regularization_losses
inon_trainable_variables

jlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?22layer_8/kernel
:22layer_8/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
kmetrics
llayer_regularization_losses
mlayer_metrics
4trainable_variables
5	variables
6regularization_losses
nnon_trainable_variables

olayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2
2layer_9/kernel
:
2layer_9/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pmetrics
qlayer_regularization_losses
rlayer_metrics
:trainable_variables
;	variables
<regularization_losses
snon_trainable_variables

tlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	wtotal
	xcount
y	variables
z	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	{total
	|count
}
_fn_kwargs
~	variables
	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
w0
x1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
{0
|1"
trackable_list_wrapper
-
~	variables"
_generic_user_object
-:+2Adam/layer_1/kernel/m
:2Adam/layer_1/bias/m
-:+2Adam/layer_3/kernel/m
:2Adam/layer_3/bias/m
':%
??2Adam/layer_7/kernel/m
 :?2Adam/layer_7/bias/m
&:$	?22Adam/layer_8/kernel/m
:22Adam/layer_8/bias/m
%:#2
2Adam/layer_9/kernel/m
:
2Adam/layer_9/bias/m
-:+2Adam/layer_1/kernel/v
:2Adam/layer_1/bias/v
-:+2Adam/layer_3/kernel/v
:2Adam/layer_3/bias/v
':%
??2Adam/layer_7/kernel/v
 :?2Adam/layer_7/bias/v
&:$	?22Adam/layer_8/kernel/v
:22Adam/layer_8/bias/v
%:#2
2Adam/layer_9/kernel/v
:
2Adam/layer_9/bias/v
?2?
__inference__wrapped_model_9036?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *4?1
/?,
layer_1_input?????????
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_9279
D__inference_sequential_layer_call_and_return_conditional_losses_9482
D__inference_sequential_layer_call_and_return_conditional_losses_9526
D__inference_sequential_layer_call_and_return_conditional_losses_9246?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_sequential_layer_call_fn_9396
)__inference_sequential_layer_call_fn_9338
)__inference_sequential_layer_call_fn_9576
)__inference_sequential_layer_call_fn_9551?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_layer_1_layer_call_and_return_conditional_losses_9587?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer_1_layer_call_fn_9596?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer_2_layer_call_and_return_conditional_losses_9042?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_layer_2_layer_call_fn_9048?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_layer_3_layer_call_and_return_conditional_losses_9607?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer_3_layer_call_fn_9616?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer_4_layer_call_and_return_conditional_losses_9054?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_layer_4_layer_call_fn_9060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_layer_5_layer_call_and_return_conditional_losses_9628
A__inference_layer_5_layer_call_and_return_conditional_losses_9633?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_layer_5_layer_call_fn_9638
&__inference_layer_5_layer_call_fn_9643?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_layer_6_layer_call_and_return_conditional_losses_9649?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer_6_layer_call_fn_9654?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer_7_layer_call_and_return_conditional_losses_9665?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer_7_layer_call_fn_9674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer_8_layer_call_and_return_conditional_losses_9685?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer_8_layer_call_fn_9694?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_layer_9_layer_call_and_return_conditional_losses_9705?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_layer_9_layer_call_fn_9714?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_9431layer_1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_9036
,-2389>?;
4?1
/?,
layer_1_input?????????
? "1?.
,
layer_9!?
layer_9?????????
?
A__inference_layer_1_layer_call_and_return_conditional_losses_9587l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_layer_1_layer_call_fn_9596_7?4
-?*
(?%
inputs?????????
? " ???????????
A__inference_layer_2_layer_call_and_return_conditional_losses_9042?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_layer_2_layer_call_fn_9048?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_layer_3_layer_call_and_return_conditional_losses_9607l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????


? ?
&__inference_layer_3_layer_call_fn_9616_7?4
-?*
(?%
inputs?????????
? " ??????????

?
A__inference_layer_4_layer_call_and_return_conditional_losses_9054?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
&__inference_layer_4_layer_call_fn_9060?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_layer_5_layer_call_and_return_conditional_losses_9628l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
A__inference_layer_5_layer_call_and_return_conditional_losses_9633l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
&__inference_layer_5_layer_call_fn_9638_;?8
1?.
(?%
inputs?????????
p
? " ???????????
&__inference_layer_5_layer_call_fn_9643_;?8
1?.
(?%
inputs?????????
p 
? " ???????????
A__inference_layer_6_layer_call_and_return_conditional_losses_9649a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ~
&__inference_layer_6_layer_call_fn_9654T7?4
-?*
(?%
inputs?????????
? "????????????
A__inference_layer_7_layer_call_and_return_conditional_losses_9665^,-0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_layer_7_layer_call_fn_9674Q,-0?-
&?#
!?
inputs??????????
? "????????????
A__inference_layer_8_layer_call_and_return_conditional_losses_9685]230?-
&?#
!?
inputs??????????
? "%?"
?
0?????????2
? z
&__inference_layer_8_layer_call_fn_9694P230?-
&?#
!?
inputs??????????
? "??????????2?
A__inference_layer_9_layer_call_and_return_conditional_losses_9705\89/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????

? y
&__inference_layer_9_layer_call_fn_9714O89/?,
%?"
 ?
inputs?????????2
? "??????????
?
D__inference_sequential_layer_call_and_return_conditional_losses_9246{
,-2389F?C
<?9
/?,
layer_1_input?????????
p

 
? "%?"
?
0?????????

? ?
D__inference_sequential_layer_call_and_return_conditional_losses_9279{
,-2389F?C
<?9
/?,
layer_1_input?????????
p 

 
? "%?"
?
0?????????

? ?
D__inference_sequential_layer_call_and_return_conditional_losses_9482t
,-2389??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
D__inference_sequential_layer_call_and_return_conditional_losses_9526t
,-2389??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
)__inference_sequential_layer_call_fn_9338n
,-2389F?C
<?9
/?,
layer_1_input?????????
p

 
? "??????????
?
)__inference_sequential_layer_call_fn_9396n
,-2389F?C
<?9
/?,
layer_1_input?????????
p 

 
? "??????????
?
)__inference_sequential_layer_call_fn_9551g
,-2389??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
)__inference_sequential_layer_call_fn_9576g
,-2389??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
"__inference_signature_wrapper_9431?
,-2389O?L
? 
E?B
@
layer_1_input/?,
layer_1_input?????????"1?.
,
layer_9!?
layer_9?????????
