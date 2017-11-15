# this code is inspired by artcg: https://gist.github.com/artcg/65b04c3fa43fab02d7ebb12e33075c32

import argparse, os

def cuda2float(filename, outname):
    f = open(filename, 'rb')
    s = f.read()
    f.close()
    CudaTensor  = b''.fromhex('10000000746F7263682E 43756461   54656E736F72  '.replace(' ', ''))
    FloatTensor = b''.fromhex('11000000746F7263682E 466C6F6174 54656E736F72  '.replace(' ', ''))
    CudaStorage = b''.fromhex('11000000746F7263682E 43756461   53746F72616765'.replace(' ', ''))
    FloatStorage= b''.fromhex('12000000746F7263682E 466C6F6174 53746F72616765'.replace(' ', ''))
    cudnnSpatialBatchNorm = b''.fromhex('1F0000006375646E6E2E5370617469616C42617463684E6F726D616C697A6174696F6E')
    nnSpatialBatchNorm = b''.fromhex('1C0000006E6E2E5370617469616C42617463684E6F726D616C697A6174696F6E')

    s = s.replace(CudaTensor, FloatTensor)
    s = s.replace(CudaStorage, FloatStorage)
    s = s.replace(cudnnSpatialBatchNorm, nnSpatialBatchNorm)

    f = open(outname, 'wb')
    f.write(s)
    f.close()

#filename = 'some_weights.t7'
#filename = '/home/ye/Developer/projects/fastSceneUnderstanding/models/fastSceneSegmentationFinal.t7'
#outname = '/home/ye/Developer/projects/fastSceneUnderstanding/models/fastSceneSegmentationFinalCPU.t7'
#cuda2float(filename, outname)

def main():
    parser = argparse.ArgumentParser(description='Convert torch t7 cuda model to cpu model')
    parser.add_argument('model',type=str, help='torch model file in t7 format')
    parser.add_argument('--output', '-o', type=str, default=None, help='output t7 file name')
    args = parser.parse_args()

    assert args.model.endswith('.t7'), 'model file must have a .t7 ext'
    output = args.output if args.output is not None else os.path.splitext(args.model)[0] + '_CPU.t7'
    cuda2float(args.model, output)

if __name__ == '__main__':
    main()
