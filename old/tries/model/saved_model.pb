¤
½
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
¼
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
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

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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¡þ
|
c2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namec2d_1/kernel
u
 c2d_1/kernel/Read/ReadVariableOpReadVariableOpc2d_1/kernel*&
_output_shapes
: *
dtype0
l

c2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
c2d_1/bias
e
c2d_1/bias/Read/ReadVariableOpReadVariableOp
c2d_1/bias*
_output_shapes
: *
dtype0
u
dns_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *
shared_namedns_1/kernel
n
 dns_1/kernel/Read/ReadVariableOpReadVariableOpdns_1/kernel*
_output_shapes
:	  *
dtype0
l

dns_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dns_1/bias
e
dns_1/bias/Read/ReadVariableOpReadVariableOp
dns_1/bias*
_output_shapes
: *
dtype0
t
dns_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedns_2/kernel
m
 dns_2/kernel/Read/ReadVariableOpReadVariableOpdns_2/kernel*
_output_shapes

: *
dtype0
l

dns_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dns_2/bias
e
dns_2/bias/Read/ReadVariableOpReadVariableOp
dns_2/bias*
_output_shapes
:*
dtype0
t
dns_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedns_3/kernel
m
 dns_3/kernel/Read/ReadVariableOpReadVariableOpdns_3/kernel*
_output_shapes

:*
dtype0
l

dns_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dns_3/bias
e
dns_3/bias/Read/ReadVariableOpReadVariableOp
dns_3/bias*
_output_shapes
:*
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

Adam/c2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/c2d_1/kernel/m

'Adam/c2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/c2d_1/kernel/m*&
_output_shapes
: *
dtype0
z
Adam/c2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/c2d_1/bias/m
s
%Adam/c2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/c2d_1/bias/m*
_output_shapes
: *
dtype0

Adam/dns_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *$
shared_nameAdam/dns_1/kernel/m
|
'Adam/dns_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dns_1/kernel/m*
_output_shapes
:	  *
dtype0
z
Adam/dns_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dns_1/bias/m
s
%Adam/dns_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dns_1/bias/m*
_output_shapes
: *
dtype0

Adam/dns_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dns_2/kernel/m
{
'Adam/dns_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dns_2/kernel/m*
_output_shapes

: *
dtype0
z
Adam/dns_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dns_2/bias/m
s
%Adam/dns_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dns_2/bias/m*
_output_shapes
:*
dtype0

Adam/dns_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dns_3/kernel/m
{
'Adam/dns_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dns_3/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dns_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dns_3/bias/m
s
%Adam/dns_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dns_3/bias/m*
_output_shapes
:*
dtype0

Adam/c2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/c2d_1/kernel/v

'Adam/c2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/c2d_1/kernel/v*&
_output_shapes
: *
dtype0
z
Adam/c2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/c2d_1/bias/v
s
%Adam/c2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/c2d_1/bias/v*
_output_shapes
: *
dtype0

Adam/dns_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	  *$
shared_nameAdam/dns_1/kernel/v
|
'Adam/dns_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dns_1/kernel/v*
_output_shapes
:	  *
dtype0
z
Adam/dns_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dns_1/bias/v
s
%Adam/dns_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dns_1/bias/v*
_output_shapes
: *
dtype0

Adam/dns_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dns_2/kernel/v
{
'Adam/dns_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dns_2/kernel/v*
_output_shapes

: *
dtype0
z
Adam/dns_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dns_2/bias/v
s
%Adam/dns_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dns_2/bias/v*
_output_shapes
:*
dtype0

Adam/dns_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dns_3/kernel/v
{
'Adam/dns_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dns_3/kernel/v*
_output_shapes

:*
dtype0
z
Adam/dns_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dns_3/bias/v
s
%Adam/dns_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dns_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
»>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ö=
valueì=Bé= Bâ=

layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
¦

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
¦

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*
¦

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses*
Ô
<iter

=beta_1

>beta_2
	?decay
@learning_ratempmq$mr%ms,mt-mu4mv5mwvxvy$vz%v{,v|-v}4v~5v*
<
0
1
$2
%3
,4
-5
46
57*
<
0
1
$2
%3
,4
-5
46
57*
* 
°
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Fserving_default* 
\V
VARIABLE_VALUEc2d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
c2d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 
* 
* 
\V
VARIABLE_VALUEdns_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dns_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdns_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dns_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdns_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dns_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

40
51*

40
51*
* 

`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

e0
f1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	gtotal
	hcount
i	variables
j	keras_api*
H
	ktotal
	lcount
m
_fn_kwargs
n	variables
o	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

g0
h1*

i	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

k0
l1*

n	variables*
y
VARIABLE_VALUEAdam/c2d_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/c2d_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dns_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dns_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dns_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dns_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dns_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dns_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/c2d_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/c2d_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dns_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dns_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dns_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dns_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dns_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dns_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_c2d_1_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ		
¬
StatefulPartitionedCallStatefulPartitionedCallserving_default_c2d_1_inputc2d_1/kernel
c2d_1/biasdns_1/kernel
dns_1/biasdns_2/kernel
dns_2/biasdns_3/kernel
dns_3/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_5060
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Û
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename c2d_1/kernel/Read/ReadVariableOpc2d_1/bias/Read/ReadVariableOp dns_1/kernel/Read/ReadVariableOpdns_1/bias/Read/ReadVariableOp dns_2/kernel/Read/ReadVariableOpdns_2/bias/Read/ReadVariableOp dns_3/kernel/Read/ReadVariableOpdns_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/c2d_1/kernel/m/Read/ReadVariableOp%Adam/c2d_1/bias/m/Read/ReadVariableOp'Adam/dns_1/kernel/m/Read/ReadVariableOp%Adam/dns_1/bias/m/Read/ReadVariableOp'Adam/dns_2/kernel/m/Read/ReadVariableOp%Adam/dns_2/bias/m/Read/ReadVariableOp'Adam/dns_3/kernel/m/Read/ReadVariableOp%Adam/dns_3/bias/m/Read/ReadVariableOp'Adam/c2d_1/kernel/v/Read/ReadVariableOp%Adam/c2d_1/bias/v/Read/ReadVariableOp'Adam/dns_1/kernel/v/Read/ReadVariableOp%Adam/dns_1/bias/v/Read/ReadVariableOp'Adam/dns_2/kernel/v/Read/ReadVariableOp%Adam/dns_2/bias/v/Read/ReadVariableOp'Adam/dns_3/kernel/v/Read/ReadVariableOp%Adam/dns_3/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
GPU 2J 8 *&
f!R
__inference__traced_save_5282
Â
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamec2d_1/kernel
c2d_1/biasdns_1/kernel
dns_1/biasdns_2/kernel
dns_2/biasdns_3/kernel
dns_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/c2d_1/kernel/mAdam/c2d_1/bias/mAdam/dns_1/kernel/mAdam/dns_1/bias/mAdam/dns_2/kernel/mAdam/dns_2/bias/mAdam/dns_3/kernel/mAdam/dns_3/bias/mAdam/c2d_1/kernel/vAdam/c2d_1/bias/vAdam/dns_1/kernel/vAdam/dns_1/bias/vAdam/dns_2/kernel/vAdam/dns_2/bias/vAdam/dns_3/kernel/vAdam/dns_3/bias/v*-
Tin&
$2"*
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_5391Åì

Ñ
D__inference_sequential_layer_call_and_return_conditional_losses_4921
c2d_1_input$

c2d_1_4898: 

c2d_1_4900: 

dns_1_4905:	  

dns_1_4907: 

dns_2_4910: 

dns_2_4912:

dns_3_4915:

dns_3_4917:
identity¢c2d_1/StatefulPartitionedCall¢dns_1/StatefulPartitionedCall¢dns_2/StatefulPartitionedCall¢dns_3/StatefulPartitionedCallë
c2d_1/StatefulPartitionedCallStatefulPartitionedCallc2d_1_input
c2d_1_4898
c2d_1_4900*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_c2d_1_layer_call_and_return_conditional_losses_4648Ø
avg_1/PartitionedCallPartitionedCall&c2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_avg_1_layer_call_and_return_conditional_losses_4628Í
flatten/PartitionedCallPartitionedCallavg_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4661ø
dns_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dns_1_4905
dns_1_4907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_1_layer_call_and_return_conditional_losses_4674þ
dns_2/StatefulPartitionedCallStatefulPartitionedCall&dns_1/StatefulPartitionedCall:output:0
dns_2_4910
dns_2_4912*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_2_layer_call_and_return_conditional_losses_4691þ
dns_3/StatefulPartitionedCallStatefulPartitionedCall&dns_2/StatefulPartitionedCall:output:0
dns_3_4915
dns_3_4917*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_3_layer_call_and_return_conditional_losses_4708u
IdentityIdentity&dns_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp^c2d_1/StatefulPartitionedCall^dns_1/StatefulPartitionedCall^dns_2/StatefulPartitionedCall^dns_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 2>
c2d_1/StatefulPartitionedCallc2d_1/StatefulPartitionedCall2>
dns_1/StatefulPartitionedCalldns_1/StatefulPartitionedCall2>
dns_2/StatefulPartitionedCalldns_2/StatefulPartitionedCall2>
dns_3/StatefulPartitionedCalldns_3/StatefulPartitionedCall:\ X
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
%
_user_specified_namec2d_1_input
·%

D__inference_sequential_layer_call_and_return_conditional_losses_5037

inputs>
$c2d_1_conv2d_readvariableop_resource: 3
%c2d_1_biasadd_readvariableop_resource: 7
$dns_1_matmul_readvariableop_resource:	  3
%dns_1_biasadd_readvariableop_resource: 6
$dns_2_matmul_readvariableop_resource: 3
%dns_2_biasadd_readvariableop_resource:6
$dns_3_matmul_readvariableop_resource:3
%dns_3_biasadd_readvariableop_resource:
identity¢c2d_1/BiasAdd/ReadVariableOp¢c2d_1/Conv2D/ReadVariableOp¢dns_1/BiasAdd/ReadVariableOp¢dns_1/MatMul/ReadVariableOp¢dns_2/BiasAdd/ReadVariableOp¢dns_2/MatMul/ReadVariableOp¢dns_3/BiasAdd/ReadVariableOp¢dns_3/MatMul/ReadVariableOp
c2d_1/Conv2D/ReadVariableOpReadVariableOp$c2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¦
c2d_1/Conv2DConv2Dinputs#c2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
~
c2d_1/BiasAdd/ReadVariableOpReadVariableOp%c2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
c2d_1/BiasAddBiasAddc2d_1/Conv2D:output:0$c2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
avg_1/AvgPoolAvgPoolc2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   }
flatten/ReshapeReshapeavg_1/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dns_1/MatMul/ReadVariableOpReadVariableOp$dns_1_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0
dns_1/MatMulMatMulflatten/Reshape:output:0#dns_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
dns_1/BiasAdd/ReadVariableOpReadVariableOp%dns_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dns_1/BiasAddBiasAdddns_1/MatMul:product:0$dns_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \

dns_1/ReluReludns_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dns_2/MatMul/ReadVariableOpReadVariableOp$dns_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dns_2/MatMulMatMuldns_1/Relu:activations:0#dns_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dns_2/BiasAdd/ReadVariableOpReadVariableOp%dns_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dns_2/BiasAddBiasAdddns_2/MatMul:product:0$dns_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

dns_2/ReluReludns_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dns_3/MatMul/ReadVariableOpReadVariableOp$dns_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dns_3/MatMulMatMuldns_2/Relu:activations:0#dns_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dns_3/BiasAdd/ReadVariableOpReadVariableOp%dns_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dns_3/BiasAddBiasAdddns_3/MatMul:product:0$dns_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dns_3/SigmoidSigmoiddns_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitydns_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^c2d_1/BiasAdd/ReadVariableOp^c2d_1/Conv2D/ReadVariableOp^dns_1/BiasAdd/ReadVariableOp^dns_1/MatMul/ReadVariableOp^dns_2/BiasAdd/ReadVariableOp^dns_2/MatMul/ReadVariableOp^dns_3/BiasAdd/ReadVariableOp^dns_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 2<
c2d_1/BiasAdd/ReadVariableOpc2d_1/BiasAdd/ReadVariableOp2:
c2d_1/Conv2D/ReadVariableOpc2d_1/Conv2D/ReadVariableOp2<
dns_1/BiasAdd/ReadVariableOpdns_1/BiasAdd/ReadVariableOp2:
dns_1/MatMul/ReadVariableOpdns_1/MatMul/ReadVariableOp2<
dns_2/BiasAdd/ReadVariableOpdns_2/BiasAdd/ReadVariableOp2:
dns_2/MatMul/ReadVariableOpdns_2/MatMul/ReadVariableOp2<
dns_3/BiasAdd/ReadVariableOpdns_3/BiasAdd/ReadVariableOp2:
dns_3/MatMul/ReadVariableOpdns_3/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
¦E
ö
__inference__traced_save_5282
file_prefix+
'savev2_c2d_1_kernel_read_readvariableop)
%savev2_c2d_1_bias_read_readvariableop+
'savev2_dns_1_kernel_read_readvariableop)
%savev2_dns_1_bias_read_readvariableop+
'savev2_dns_2_kernel_read_readvariableop)
%savev2_dns_2_bias_read_readvariableop+
'savev2_dns_3_kernel_read_readvariableop)
%savev2_dns_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_c2d_1_kernel_m_read_readvariableop0
,savev2_adam_c2d_1_bias_m_read_readvariableop2
.savev2_adam_dns_1_kernel_m_read_readvariableop0
,savev2_adam_dns_1_bias_m_read_readvariableop2
.savev2_adam_dns_2_kernel_m_read_readvariableop0
,savev2_adam_dns_2_bias_m_read_readvariableop2
.savev2_adam_dns_3_kernel_m_read_readvariableop0
,savev2_adam_dns_3_bias_m_read_readvariableop2
.savev2_adam_c2d_1_kernel_v_read_readvariableop0
,savev2_adam_c2d_1_bias_v_read_readvariableop2
.savev2_adam_dns_1_kernel_v_read_readvariableop0
,savev2_adam_dns_1_bias_v_read_readvariableop2
.savev2_adam_dns_2_kernel_v_read_readvariableop0
,savev2_adam_dns_2_bias_v_read_readvariableop2
.savev2_adam_dns_3_kernel_v_read_readvariableop0
,savev2_adam_dns_3_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¯
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ø
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ù
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_c2d_1_kernel_read_readvariableop%savev2_c2d_1_bias_read_readvariableop'savev2_dns_1_kernel_read_readvariableop%savev2_dns_1_bias_read_readvariableop'savev2_dns_2_kernel_read_readvariableop%savev2_dns_2_bias_read_readvariableop'savev2_dns_3_kernel_read_readvariableop%savev2_dns_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_c2d_1_kernel_m_read_readvariableop,savev2_adam_c2d_1_bias_m_read_readvariableop.savev2_adam_dns_1_kernel_m_read_readvariableop,savev2_adam_dns_1_bias_m_read_readvariableop.savev2_adam_dns_2_kernel_m_read_readvariableop,savev2_adam_dns_2_bias_m_read_readvariableop.savev2_adam_dns_3_kernel_m_read_readvariableop,savev2_adam_dns_3_bias_m_read_readvariableop.savev2_adam_c2d_1_kernel_v_read_readvariableop,savev2_adam_c2d_1_bias_v_read_readvariableop.savev2_adam_dns_1_kernel_v_read_readvariableop,savev2_adam_dns_1_bias_v_read_readvariableop.savev2_adam_dns_2_kernel_v_read_readvariableop,savev2_adam_dns_2_bias_v_read_readvariableop.savev2_adam_dns_3_kernel_v_read_readvariableop,savev2_adam_dns_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesô
ñ: : : :	  : : :::: : : : : : : : : : : :	  : : :::: : :	  : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :
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
: :,(
&
_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::"

_output_shapes
: 
æ	
Æ
)__inference_sequential_layer_call_fn_4734
c2d_1_input!
unknown: 
	unknown_0: 
	unknown_1:	  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallc2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4715o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
%
_user_specified_namec2d_1_input

Ñ
D__inference_sequential_layer_call_and_return_conditional_losses_4895
c2d_1_input$

c2d_1_4872: 

c2d_1_4874: 

dns_1_4879:	  

dns_1_4881: 

dns_2_4884: 

dns_2_4886:

dns_3_4889:

dns_3_4891:
identity¢c2d_1/StatefulPartitionedCall¢dns_1/StatefulPartitionedCall¢dns_2/StatefulPartitionedCall¢dns_3/StatefulPartitionedCallë
c2d_1/StatefulPartitionedCallStatefulPartitionedCallc2d_1_input
c2d_1_4872
c2d_1_4874*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_c2d_1_layer_call_and_return_conditional_losses_4648Ø
avg_1/PartitionedCallPartitionedCall&c2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_avg_1_layer_call_and_return_conditional_losses_4628Í
flatten/PartitionedCallPartitionedCallavg_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4661ø
dns_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dns_1_4879
dns_1_4881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_1_layer_call_and_return_conditional_losses_4674þ
dns_2/StatefulPartitionedCallStatefulPartitionedCall&dns_1/StatefulPartitionedCall:output:0
dns_2_4884
dns_2_4886*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_2_layer_call_and_return_conditional_losses_4691þ
dns_3/StatefulPartitionedCallStatefulPartitionedCall&dns_2/StatefulPartitionedCall:output:0
dns_3_4889
dns_3_4891*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_3_layer_call_and_return_conditional_losses_4708u
IdentityIdentity&dns_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp^c2d_1/StatefulPartitionedCall^dns_1/StatefulPartitionedCall^dns_2/StatefulPartitionedCall^dns_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 2>
c2d_1/StatefulPartitionedCallc2d_1/StatefulPartitionedCall2>
dns_1/StatefulPartitionedCalldns_1/StatefulPartitionedCall2>
dns_2/StatefulPartitionedCalldns_2/StatefulPartitionedCall2>
dns_3/StatefulPartitionedCalldns_3/StatefulPartitionedCall:\ X
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
%
_user_specified_namec2d_1_input


ñ
?__inference_dns_1_layer_call_and_return_conditional_losses_4674

inputs1
matmul_readvariableop_resource:	  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

[
?__inference_avg_1_layer_call_and_return_conditional_losses_5089

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
B
&__inference_flatten_layer_call_fn_5094

inputs
identity­
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4661a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ã
]
A__inference_flatten_layer_call_and_return_conditional_losses_4661

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
à

$__inference_c2d_1_layer_call_fn_5069

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_c2d_1_layer_call_and_return_conditional_losses_4648w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ		: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
£

ø
?__inference_c2d_1_layer_call_and_return_conditional_losses_5079

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs


ð
?__inference_dns_2_layer_call_and_return_conditional_losses_4691

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


ñ
?__inference_dns_1_layer_call_and_return_conditional_losses_5120

inputs1
matmul_readvariableop_resource:	  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


 __inference__traced_restore_5391
file_prefix7
assignvariableop_c2d_1_kernel: +
assignvariableop_1_c2d_1_bias: 2
assignvariableop_2_dns_1_kernel:	  +
assignvariableop_3_dns_1_bias: 1
assignvariableop_4_dns_2_kernel: +
assignvariableop_5_dns_2_bias:1
assignvariableop_6_dns_3_kernel:+
assignvariableop_7_dns_3_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: A
'assignvariableop_17_adam_c2d_1_kernel_m: 3
%assignvariableop_18_adam_c2d_1_bias_m: :
'assignvariableop_19_adam_dns_1_kernel_m:	  3
%assignvariableop_20_adam_dns_1_bias_m: 9
'assignvariableop_21_adam_dns_2_kernel_m: 3
%assignvariableop_22_adam_dns_2_bias_m:9
'assignvariableop_23_adam_dns_3_kernel_m:3
%assignvariableop_24_adam_dns_3_bias_m:A
'assignvariableop_25_adam_c2d_1_kernel_v: 3
%assignvariableop_26_adam_c2d_1_bias_v: :
'assignvariableop_27_adam_dns_1_kernel_v:	  3
%assignvariableop_28_adam_dns_1_bias_v: 9
'assignvariableop_29_adam_dns_2_kernel_v: 3
%assignvariableop_30_adam_dns_2_bias_v:9
'assignvariableop_31_adam_dns_3_kernel_v:3
%assignvariableop_32_adam_dns_3_bias_v:
identity_34¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9²
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ø
valueÎBË"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_c2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_c2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_dns_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dns_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dns_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dns_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_dns_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dns_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_c2d_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_c2d_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dns_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dns_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dns_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dns_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dns_3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dns_3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_c2d_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_c2d_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dns_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dns_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dns_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp%assignvariableop_30_adam_dns_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dns_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dns_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¥
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322(
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


ð
?__inference_dns_2_layer_call_and_return_conditional_losses_5140

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


ð
?__inference_dns_3_layer_call_and_return_conditional_losses_5160

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×	
Á
)__inference_sequential_layer_call_fn_4948

inputs!
unknown: 
	unknown_0: 
	unknown_1:	  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4715o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs


ð
?__inference_dns_3_layer_call_and_return_conditional_losses_4708

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×	
Á
)__inference_sequential_layer_call_fn_4969

inputs!
unknown: 
	unknown_0: 
	unknown_1:	  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4829o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs

Ì
D__inference_sequential_layer_call_and_return_conditional_losses_4715

inputs$

c2d_1_4649: 

c2d_1_4651: 

dns_1_4675:	  

dns_1_4677: 

dns_2_4692: 

dns_2_4694:

dns_3_4709:

dns_3_4711:
identity¢c2d_1/StatefulPartitionedCall¢dns_1/StatefulPartitionedCall¢dns_2/StatefulPartitionedCall¢dns_3/StatefulPartitionedCallæ
c2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs
c2d_1_4649
c2d_1_4651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_c2d_1_layer_call_and_return_conditional_losses_4648Ø
avg_1/PartitionedCallPartitionedCall&c2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_avg_1_layer_call_and_return_conditional_losses_4628Í
flatten/PartitionedCallPartitionedCallavg_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4661ø
dns_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dns_1_4675
dns_1_4677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_1_layer_call_and_return_conditional_losses_4674þ
dns_2/StatefulPartitionedCallStatefulPartitionedCall&dns_1/StatefulPartitionedCall:output:0
dns_2_4692
dns_2_4694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_2_layer_call_and_return_conditional_losses_4691þ
dns_3/StatefulPartitionedCallStatefulPartitionedCall&dns_2/StatefulPartitionedCall:output:0
dns_3_4709
dns_3_4711*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_3_layer_call_and_return_conditional_losses_4708u
IdentityIdentity&dns_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp^c2d_1/StatefulPartitionedCall^dns_1/StatefulPartitionedCall^dns_2/StatefulPartitionedCall^dns_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 2>
c2d_1/StatefulPartitionedCallc2d_1/StatefulPartitionedCall2>
dns_1/StatefulPartitionedCalldns_1/StatefulPartitionedCall2>
dns_2/StatefulPartitionedCalldns_2/StatefulPartitionedCall2>
dns_3/StatefulPartitionedCalldns_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
 
@
$__inference_avg_1_layer_call_fn_5084

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_avg_1_layer_call_and_return_conditional_losses_4628
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»

$__inference_dns_1_layer_call_fn_5109

inputs
unknown:	  
	unknown_0: 
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_1_layer_call_and_return_conditional_losses_4674o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¸

$__inference_dns_2_layer_call_fn_5129

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_2_layer_call_and_return_conditional_losses_4691o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¸

$__inference_dns_3_layer_call_fn_5149

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_3_layer_call_and_return_conditional_losses_4708o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

[
?__inference_avg_1_layer_call_and_return_conditional_losses_4628

inputs
identity«
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·%

D__inference_sequential_layer_call_and_return_conditional_losses_5003

inputs>
$c2d_1_conv2d_readvariableop_resource: 3
%c2d_1_biasadd_readvariableop_resource: 7
$dns_1_matmul_readvariableop_resource:	  3
%dns_1_biasadd_readvariableop_resource: 6
$dns_2_matmul_readvariableop_resource: 3
%dns_2_biasadd_readvariableop_resource:6
$dns_3_matmul_readvariableop_resource:3
%dns_3_biasadd_readvariableop_resource:
identity¢c2d_1/BiasAdd/ReadVariableOp¢c2d_1/Conv2D/ReadVariableOp¢dns_1/BiasAdd/ReadVariableOp¢dns_1/MatMul/ReadVariableOp¢dns_2/BiasAdd/ReadVariableOp¢dns_2/MatMul/ReadVariableOp¢dns_3/BiasAdd/ReadVariableOp¢dns_3/MatMul/ReadVariableOp
c2d_1/Conv2D/ReadVariableOpReadVariableOp$c2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¦
c2d_1/Conv2DConv2Dinputs#c2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
~
c2d_1/BiasAdd/ReadVariableOpReadVariableOp%c2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
c2d_1/BiasAddBiasAddc2d_1/Conv2D:output:0$c2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¦
avg_1/AvgPoolAvgPoolc2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   }
flatten/ReshapeReshapeavg_1/AvgPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dns_1/MatMul/ReadVariableOpReadVariableOp$dns_1_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0
dns_1/MatMulMatMulflatten/Reshape:output:0#dns_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ~
dns_1/BiasAdd/ReadVariableOpReadVariableOp%dns_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
dns_1/BiasAddBiasAdddns_1/MatMul:product:0$dns_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ \

dns_1/ReluReludns_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dns_2/MatMul/ReadVariableOpReadVariableOp$dns_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dns_2/MatMulMatMuldns_1/Relu:activations:0#dns_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dns_2/BiasAdd/ReadVariableOpReadVariableOp%dns_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dns_2/BiasAddBiasAdddns_2/MatMul:product:0$dns_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

dns_2/ReluReludns_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dns_3/MatMul/ReadVariableOpReadVariableOp$dns_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dns_3/MatMulMatMuldns_2/Relu:activations:0#dns_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dns_3/BiasAdd/ReadVariableOpReadVariableOp%dns_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dns_3/BiasAddBiasAdddns_3/MatMul:product:0$dns_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dns_3/SigmoidSigmoiddns_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitydns_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^c2d_1/BiasAdd/ReadVariableOp^c2d_1/Conv2D/ReadVariableOp^dns_1/BiasAdd/ReadVariableOp^dns_1/MatMul/ReadVariableOp^dns_2/BiasAdd/ReadVariableOp^dns_2/MatMul/ReadVariableOp^dns_3/BiasAdd/ReadVariableOp^dns_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 2<
c2d_1/BiasAdd/ReadVariableOpc2d_1/BiasAdd/ReadVariableOp2:
c2d_1/Conv2D/ReadVariableOpc2d_1/Conv2D/ReadVariableOp2<
dns_1/BiasAdd/ReadVariableOpdns_1/BiasAdd/ReadVariableOp2:
dns_1/MatMul/ReadVariableOpdns_1/MatMul/ReadVariableOp2<
dns_2/BiasAdd/ReadVariableOpdns_2/BiasAdd/ReadVariableOp2:
dns_2/MatMul/ReadVariableOpdns_2/MatMul/ReadVariableOp2<
dns_3/BiasAdd/ReadVariableOpdns_3/BiasAdd/ReadVariableOp2:
dns_3/MatMul/ReadVariableOpdns_3/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
º	
¿
"__inference_signature_wrapper_5060
c2d_1_input!
unknown: 
	unknown_0: 
	unknown_1:	  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallc2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_4619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
%
_user_specified_namec2d_1_input
-
¤
__inference__wrapped_model_4619
c2d_1_inputI
/sequential_c2d_1_conv2d_readvariableop_resource: >
0sequential_c2d_1_biasadd_readvariableop_resource: B
/sequential_dns_1_matmul_readvariableop_resource:	  >
0sequential_dns_1_biasadd_readvariableop_resource: A
/sequential_dns_2_matmul_readvariableop_resource: >
0sequential_dns_2_biasadd_readvariableop_resource:A
/sequential_dns_3_matmul_readvariableop_resource:>
0sequential_dns_3_biasadd_readvariableop_resource:
identity¢'sequential/c2d_1/BiasAdd/ReadVariableOp¢&sequential/c2d_1/Conv2D/ReadVariableOp¢'sequential/dns_1/BiasAdd/ReadVariableOp¢&sequential/dns_1/MatMul/ReadVariableOp¢'sequential/dns_2/BiasAdd/ReadVariableOp¢&sequential/dns_2/MatMul/ReadVariableOp¢'sequential/dns_3/BiasAdd/ReadVariableOp¢&sequential/dns_3/MatMul/ReadVariableOp
&sequential/c2d_1/Conv2D/ReadVariableOpReadVariableOp/sequential_c2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Á
sequential/c2d_1/Conv2DConv2Dc2d_1_input.sequential/c2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

'sequential/c2d_1/BiasAdd/ReadVariableOpReadVariableOp0sequential_c2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
sequential/c2d_1/BiasAddBiasAdd sequential/c2d_1/Conv2D:output:0/sequential/c2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¼
sequential/avg_1/AvgPoolAvgPool!sequential/c2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
sequential/flatten/ReshapeReshape!sequential/avg_1/AvgPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&sequential/dns_1/MatMul/ReadVariableOpReadVariableOp/sequential_dns_1_matmul_readvariableop_resource*
_output_shapes
:	  *
dtype0¨
sequential/dns_1/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dns_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'sequential/dns_1/BiasAdd/ReadVariableOpReadVariableOp0sequential_dns_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0©
sequential/dns_1/BiasAddBiasAdd!sequential/dns_1/MatMul:product:0/sequential/dns_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
sequential/dns_1/ReluRelu!sequential/dns_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
&sequential/dns_2/MatMul/ReadVariableOpReadVariableOp/sequential_dns_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0¨
sequential/dns_2/MatMulMatMul#sequential/dns_1/Relu:activations:0.sequential/dns_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dns_2/BiasAdd/ReadVariableOpReadVariableOp0sequential_dns_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
sequential/dns_2/BiasAddBiasAdd!sequential/dns_2/MatMul:product:0/sequential/dns_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
sequential/dns_2/ReluRelu!sequential/dns_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential/dns_3/MatMul/ReadVariableOpReadVariableOp/sequential_dns_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¨
sequential/dns_3/MatMulMatMul#sequential/dns_2/Relu:activations:0.sequential/dns_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dns_3/BiasAdd/ReadVariableOpReadVariableOp0sequential_dns_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
sequential/dns_3/BiasAddBiasAdd!sequential/dns_3/MatMul:product:0/sequential/dns_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
sequential/dns_3/SigmoidSigmoid!sequential/dns_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentitysequential/dns_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp(^sequential/c2d_1/BiasAdd/ReadVariableOp'^sequential/c2d_1/Conv2D/ReadVariableOp(^sequential/dns_1/BiasAdd/ReadVariableOp'^sequential/dns_1/MatMul/ReadVariableOp(^sequential/dns_2/BiasAdd/ReadVariableOp'^sequential/dns_2/MatMul/ReadVariableOp(^sequential/dns_3/BiasAdd/ReadVariableOp'^sequential/dns_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 2R
'sequential/c2d_1/BiasAdd/ReadVariableOp'sequential/c2d_1/BiasAdd/ReadVariableOp2P
&sequential/c2d_1/Conv2D/ReadVariableOp&sequential/c2d_1/Conv2D/ReadVariableOp2R
'sequential/dns_1/BiasAdd/ReadVariableOp'sequential/dns_1/BiasAdd/ReadVariableOp2P
&sequential/dns_1/MatMul/ReadVariableOp&sequential/dns_1/MatMul/ReadVariableOp2R
'sequential/dns_2/BiasAdd/ReadVariableOp'sequential/dns_2/BiasAdd/ReadVariableOp2P
&sequential/dns_2/MatMul/ReadVariableOp&sequential/dns_2/MatMul/ReadVariableOp2R
'sequential/dns_3/BiasAdd/ReadVariableOp'sequential/dns_3/BiasAdd/ReadVariableOp2P
&sequential/dns_3/MatMul/ReadVariableOp&sequential/dns_3/MatMul/ReadVariableOp:\ X
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
%
_user_specified_namec2d_1_input
£

ø
?__inference_c2d_1_layer_call_and_return_conditional_losses_4648

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ		: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
Ã
]
A__inference_flatten_layer_call_and_return_conditional_losses_5100

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

Ì
D__inference_sequential_layer_call_and_return_conditional_losses_4829

inputs$

c2d_1_4806: 

c2d_1_4808: 

dns_1_4813:	  

dns_1_4815: 

dns_2_4818: 

dns_2_4820:

dns_3_4823:

dns_3_4825:
identity¢c2d_1/StatefulPartitionedCall¢dns_1/StatefulPartitionedCall¢dns_2/StatefulPartitionedCall¢dns_3/StatefulPartitionedCallæ
c2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs
c2d_1_4806
c2d_1_4808*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_c2d_1_layer_call_and_return_conditional_losses_4648Ø
avg_1/PartitionedCallPartitionedCall&c2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_avg_1_layer_call_and_return_conditional_losses_4628Í
flatten/PartitionedCallPartitionedCallavg_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_4661ø
dns_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
dns_1_4813
dns_1_4815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_1_layer_call_and_return_conditional_losses_4674þ
dns_2/StatefulPartitionedCallStatefulPartitionedCall&dns_1/StatefulPartitionedCall:output:0
dns_2_4818
dns_2_4820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_2_layer_call_and_return_conditional_losses_4691þ
dns_3/StatefulPartitionedCallStatefulPartitionedCall&dns_2/StatefulPartitionedCall:output:0
dns_3_4823
dns_3_4825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dns_3_layer_call_and_return_conditional_losses_4708u
IdentityIdentity&dns_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp^c2d_1/StatefulPartitionedCall^dns_1/StatefulPartitionedCall^dns_2/StatefulPartitionedCall^dns_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 2>
c2d_1/StatefulPartitionedCallc2d_1/StatefulPartitionedCall2>
dns_1/StatefulPartitionedCalldns_1/StatefulPartitionedCall2>
dns_2/StatefulPartitionedCalldns_2/StatefulPartitionedCall2>
dns_3/StatefulPartitionedCalldns_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
 
_user_specified_nameinputs
æ	
Æ
)__inference_sequential_layer_call_fn_4869
c2d_1_input!
unknown: 
	unknown_0: 
	unknown_1:	  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallc2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4829o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ		: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ		
%
_user_specified_namec2d_1_input"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¸
serving_default¤
K
c2d_1_input<
serving_default_c2d_1_input:0ÿÿÿÿÿÿÿÿÿ		9
dns_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Òu

layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
»

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
»

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
»

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses"
_tf_keras_layer
ã
<iter

=beta_1

>beta_2
	?decay
@learning_ratempmq$mr%ms,mt-mu4mv5mwvxvy$vz%v{,v|-v}4v~5v"
	optimizer
X
0
1
$2
%3
,4
-5
46
57"
trackable_list_wrapper
X
0
1
$2
%3
,4
-5
46
57"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_sequential_layer_call_fn_4734
)__inference_sequential_layer_call_fn_4948
)__inference_sequential_layer_call_fn_4969
)__inference_sequential_layer_call_fn_4869À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_sequential_layer_call_and_return_conditional_losses_5003
D__inference_sequential_layer_call_and_return_conditional_losses_5037
D__inference_sequential_layer_call_and_return_conditional_losses_4895
D__inference_sequential_layer_call_and_return_conditional_losses_4921À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÎBË
__inference__wrapped_model_4619c2d_1_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Fserving_default"
signature_map
&:$ 2c2d_1/kernel
: 2
c2d_1/bias
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
­
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_c2d_1_layer_call_fn_5069¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_c2d_1_layer_call_and_return_conditional_losses_5079¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_avg_1_layer_call_fn_5084¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_avg_1_layer_call_and_return_conditional_losses_5089¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_flatten_layer_call_fn_5094¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_flatten_layer_call_and_return_conditional_losses_5100¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	  2dns_1/kernel
: 2
dns_1/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_dns_1_layer_call_fn_5109¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_dns_1_layer_call_and_return_conditional_losses_5120¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
: 2dns_2/kernel
:2
dns_2/bias
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
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_dns_2_layer_call_fn_5129¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_dns_2_layer_call_and_return_conditional_losses_5140¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:2dns_3/kernel
:2
dns_3/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
­
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
Î2Ë
$__inference_dns_3_layer_call_fn_5149¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_dns_3_layer_call_and_return_conditional_losses_5160¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÍBÊ
"__inference_signature_wrapper_5060c2d_1_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
N
	gtotal
	hcount
i	variables
j	keras_api"
_tf_keras_metric
^
	ktotal
	lcount
m
_fn_kwargs
n	variables
o	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
g0
h1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
k0
l1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
+:) 2Adam/c2d_1/kernel/m
: 2Adam/c2d_1/bias/m
$:"	  2Adam/dns_1/kernel/m
: 2Adam/dns_1/bias/m
#:! 2Adam/dns_2/kernel/m
:2Adam/dns_2/bias/m
#:!2Adam/dns_3/kernel/m
:2Adam/dns_3/bias/m
+:) 2Adam/c2d_1/kernel/v
: 2Adam/c2d_1/bias/v
$:"	  2Adam/dns_1/kernel/v
: 2Adam/dns_1/bias/v
#:! 2Adam/dns_2/kernel/v
:2Adam/dns_2/bias/v
#:!2Adam/dns_3/kernel/v
:2Adam/dns_3/bias/v
__inference__wrapped_model_4619w$%,-45<¢9
2¢/
-*
c2d_1_inputÿÿÿÿÿÿÿÿÿ		
ª "-ª*
(
dns_3
dns_3ÿÿÿÿÿÿÿÿÿâ
?__inference_avg_1_layer_call_and_return_conditional_losses_5089R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
$__inference_avg_1_layer_call_fn_5084R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
?__inference_c2d_1_layer_call_and_return_conditional_losses_5079l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
$__inference_c2d_1_layer_call_fn_5069_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ		
ª " ÿÿÿÿÿÿÿÿÿ  
?__inference_dns_1_layer_call_and_return_conditional_losses_5120]$%0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 x
$__inference_dns_1_layer_call_fn_5109P$%0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ 
?__inference_dns_2_layer_call_and_return_conditional_losses_5140\,-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
$__inference_dns_2_layer_call_fn_5129O,-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ
?__inference_dns_3_layer_call_and_return_conditional_losses_5160\45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
$__inference_dns_3_layer_call_fn_5149O45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
A__inference_flatten_layer_call_and_return_conditional_losses_5100a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ 
 ~
&__inference_flatten_layer_call_fn_5094T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ¿
D__inference_sequential_layer_call_and_return_conditional_losses_4895w$%,-45D¢A
:¢7
-*
c2d_1_inputÿÿÿÿÿÿÿÿÿ		
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
D__inference_sequential_layer_call_and_return_conditional_losses_4921w$%,-45D¢A
:¢7
-*
c2d_1_inputÿÿÿÿÿÿÿÿÿ		
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
D__inference_sequential_layer_call_and_return_conditional_losses_5003r$%,-45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ		
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
D__inference_sequential_layer_call_and_return_conditional_losses_5037r$%,-45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ		
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_sequential_layer_call_fn_4734j$%,-45D¢A
:¢7
-*
c2d_1_inputÿÿÿÿÿÿÿÿÿ		
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_4869j$%,-45D¢A
:¢7
-*
c2d_1_inputÿÿÿÿÿÿÿÿÿ		
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_4948e$%,-45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ		
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_4969e$%,-45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ		
p

 
ª "ÿÿÿÿÿÿÿÿÿ­
"__inference_signature_wrapper_5060$%,-45K¢H
¢ 
Aª>
<
c2d_1_input-*
c2d_1_inputÿÿÿÿÿÿÿÿÿ		"-ª*
(
dns_3
dns_3ÿÿÿÿÿÿÿÿÿ