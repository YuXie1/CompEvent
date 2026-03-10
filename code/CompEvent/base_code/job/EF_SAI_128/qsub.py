import os
import os.path as osp
import sys
import argparse

GPU_dict = {
    '1080Ti': {1: '1:S', 2: '2:D', 4: '4:Q'},
    'K80': {1: '1:s', 2: '2:d', 4: '4:q', 8: '8:E'},
    '2080Ti': {1: '1:s', 2: '2:d', 4: '4:q', 8: '8:E'},
    '3090': {1: '1:A', 2: '2:B', 4: '4:C', 8: '8:F'},
}

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default=None, help="job name")
parser.add_argument('--image', default='bit:5000/caocz_py36_torch1.7', type=str, help='image name')
parser.add_argument('--sub', action='store_true', help='qsub')
parser.add_argument('--gpu', default=None, type=str, help='gpu text')
parser.add_argument('--gpu-num', type=int, default=1, help='gpu_num: 1  2  4')
parser.add_argument('--gpu-type', default='2080Ti', type=str, help='gpu_type: e.g. 1080Ti  K80')
opt = parser.parse_args()

job_dir = os.getcwd()
_, default_name = osp.split(job_dir)
file_path = osp.join(job_dir, 'job.pbs')
assert osp.isfile(osp.join(job_dir, 'start'))
os.system('chmod +x {}'.format(osp.join(job_dir, 'start')))

job_name = opt.name if opt.name is not None else default_name
image_name = opt.image
if opt.gpu:
    gpu_text = opt.gpu
else:
    gpu_text = GPU_dict[opt.gpu_type][opt.gpu_num]

text = \
'''\
#PBS    -N  {0}
#PBS    -o  {1}/out.out
#PBS    -e  {1}/err.err
#PBS    -l  nodes=1:gpus={2}
#PBS    -r  y
cd $PBS_O_WORKDIR
echo Time is `date`
echo Job name is ${{PBS_JOBNAME}}
echo Directory is $PWD 
echo CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES}}
echo This job runs on following nodes:
cat $PBS_NODEFILE
cat $PBS_GPUFILE

startdocker -D /gdata/linrj/ -P /ghome/linrj/ -u "-v /gdata1/linrj/:/gdata1/linrj/ -e PYTHONPATH=/gdata/linrj/pylib --shm-size=8gb" -c "{1}/start" {3}\
'''.format(job_name, job_dir, gpu_text, image_name)

with open(file_path, 'w') as f:
    f.write(text + '\n')

print('-'*50)
print(text)
print('-'*50)

if opt.sub:
    os.system('qsub {}'.format(file_path))
