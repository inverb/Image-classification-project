?$	?lXSY`_@?lXSY`_@!?lXSY`_@	?7]???I@?7]???I@!?7]???I@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?lXSY`_@??????F@1?*Q??
+@A?jGq?:??I??O?Y???YH0?[P@*e;?O??u@???M?o?@2j
3Iterator::Model::MaxIntraOpParallelism::Filter::Map??p?!N@!&?n$??M@)???
N@1hM?@??M@:Preprocessing2?
LIterator::Model::MaxIntraOpParallelism::Filter::Map::Prefetch::PaddedBatchV2Ɏ?@?n2@!%???Y2@)?U??yL2@1r6}e?72@:Preprocessing2?
jIterator::Model::MaxIntraOpParallelism::Filter::Map::Prefetch::PaddedBatchV2::ParallelMapV2::ParallelMapV2??Ia?"@!m????!@)??Ia?"@1m????!@:Preprocessing2?
}Iterator::Model::MaxIntraOpParallelism::Filter::Map::Prefetch::PaddedBatchV2::ParallelMapV2::ParallelMapV2::AssertCardinality?~K??!@!??Z?!@):?,B??!@1<???ґ!@:Preprocessing2e
.Iterator::Model::MaxIntraOpParallelism::Filter0?^|ѶO@!,Ez??O@)x*???O	@1d`? h3	@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Filter::Map::Prefetch::PaddedBatchV2::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap[0]::TFRecordZf?????!?oiV????)Zf?????1?oiV????:Advanced file read2?
?Iterator::Model::MaxIntraOpParallelism::Filter::Map::Prefetch::PaddedBatchV2::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap?8?Zn??!aC???[??)??x??1.??BZ??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Filter::Map::PrefetchS	O??'??!???????)S	O??'??1???????:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Filter::Map::Prefetch::PaddedBatchV2::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4X?l:???!??7?Ρ??)X?l:???1??7?Ρ??:Preprocessing2?
[Iterator::Model::MaxIntraOpParallelism::Filter::Map::Prefetch::PaddedBatchV2::ParallelMapV2???hW!??!qYs??)???hW!??1qYs??:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Filter::Map::Prefetch::PaddedBatchV2::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[2]::FlatMap[0]::TFRecord?Y?????!?ЩBX~??)?Y?????1?ЩBX~??:Advanced file read2?
?Iterator::Model::MaxIntraOpParallelism::Filter::Map::Prefetch::PaddedBatchV2::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[2]::FlatMapz?'L??!?????)?J[\?3??1?y\I???:Preprocessing2F
Iterator::Model???[˸O@!B+AJ?O@)Y???j??1 ݱ??V??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??????O@!$E???O@)??s?f|?1?_Y?F|?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 51.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t36.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?7]???I@IԵ?O?C@QzJĢċ%@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????F@??????F@!??????F@      ??!       "	?*Q??
+@?*Q??
+@!?*Q??
+@*      ??!       2	?jGq?:???jGq?:??!?jGq?:??:	??O?Y?????O?Y???!??O?Y???B      ??!       J	H0?[P@H0?[P@!H0?[P@R      ??!       Z	H0?[P@H0?[P@!H0?[P@b      ??!       JGPUY?7]???I@b qԵ?O?C@yzJĢċ%@?"K
$Adam/Adam/update_6/ResourceApplyAdamResourceApplyAdam???a??!???a??"G
+gradient_tape/sequential_3/dense_6/MatMul_1MatMul????`???!????_??"-
IteratorGetNext/_1_Send@O/?@???!??6Ι???"G
)gradient_tape/sequential_3/dense_6/MatMulMatMul,??"??!/?c?????0"9
sequential_3/dense_6/MatMulMatMul?,?????!??/????0"k
?gradient_tape/sequential_3/conv2d_9/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?#?%҆??!4??u+??0"<
sequential_3/conv2d_9/Relu_FusedConv2D??Dsu??!???? ???"l
@gradient_tape/sequential_3/conv2d_10/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?#??ۈ??!??n?g???0"j
?gradient_tape/sequential_3/conv2d_10/Conv2D/Conv2DBackpropInputConv2DBackpropInput???8O???!??Q?,???0"_
>gradient_tape/sequential_3/max_pooling2d_6/MaxPool/MaxPoolGradMaxPoolGrad?????!???z;??Q      Y@Y:?2ϵR@a?5??(9@q?Ǎ???y??s#???"?

host?Your program is HIGHLY input-bound because 51.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t36.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 