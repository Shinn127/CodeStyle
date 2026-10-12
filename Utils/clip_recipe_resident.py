"""Immutable train features/code indices on device; CPU sampling order is unchanged."""
import numpy as np
import torch


class ResidentTrainingData:
    def __init__(self,dataset,cache,mean,std,*,device,budget_gb=10.,reserve_gb=2.,decoder=None):
        self.device=torch.device(device);self.features=None;self.indices=None;self.teacher_motion=None
        self.segment_offsets=np.full(len(dataset.segments),-1,np.int64)
        self.info=dict(mode='cpu',resident_bytes=0)
        if budget_gb<=0:return
        train=[(i,s) for i,s in enumerate(dataset.segments) if s['split']=='train']
        count=sum(s['num_frames'] for _,s in train)
        index_bytes=cache.meta.num_frames*cache.meta.num_codebooks*4
        feature_bytes=count*375*4
        capacity=int(budget_gb*2**30)
        if self.device.type=='cuda':
            reusable=torch.cuda.memory_reserved(self.device)-torch.cuda.memory_allocated(self.device)
            capacity=min(capacity,max(0,torch.cuda.mem_get_info(self.device)[0]+reusable-int(reserve_gb*2**30)))
        if capacity<index_bytes:return
        try:self.indices=torch.empty(cache.indices.shape,device=self.device,dtype=torch.int32)
        except torch.OutOfMemoryError:
            self.info['fallback']='index_allocation_oom'
            return
        for start in range(0,cache.meta.num_frames,65536):
            self.indices[start:start+65536].copy_(torch.from_numpy(np.array(cache.indices[start:start+65536],dtype=np.int32)))
        self.info=dict(mode='indices',resident_bytes=index_bytes)
        if capacity<index_bytes+feature_bytes:return
        try:self.features=torch.empty((count,375),device=self.device,dtype=torch.float32)
        except torch.OutOfMemoryError:
            self.info['fallback']='feature_allocation_oom'
            return
        source=np.memmap(dataset.db_path/'X.f32',mode='r',dtype=np.float32,shape=(cache.meta.num_frames,375))
        offset=0
        for index,s in train:
            self.segment_offsets[index]=offset
            for local in range(0,s['num_frames'],32768):
                size=min(32768,s['num_frames']-local)
                x=np.array(source[s['frame_start']+local:s['frame_start']+local+size],copy=True)
                # Identical two NumPy FP32 operations to Motion375Dataset normalization.
                np.subtract(x,mean,out=x);np.divide(x,std,out=x)
                self.features[offset+local:offset+local+size].copy_(torch.from_numpy(x))
            offset+=s['num_frames']
        self.info=dict(mode='train_features_and_indices',resident_bytes=index_bytes+feature_bytes,
            train_frames=count,capacity_bytes=capacity)
        motion_bytes=count*291*4
        if decoder is not None and index_bytes+feature_bytes+motion_bytes>capacity:
            self.info['codec_fallback']='budget_or_free_memory'
        if decoder is not None and index_bytes+feature_bytes+motion_bytes<=capacity:
            try:self.teacher_motion=torch.empty((count,291),device=self.device,dtype=torch.float32)
            except torch.OutOfMemoryError:
                self.info['codec_fallback']='allocation_oom'
                return
            with torch.no_grad():
                for index,s in train:
                    offset=self.segment_offsets[index]
                    for local in range(0,s['num_frames'],8192):
                        size=min(8192,s['num_frames']-local)
                        codes=self.indices[s['frame_start']+local:s['frame_start']+local+size].long()
                        decoded=decoder.decode(decoder.quantizer.lookup(codes))
                        self.teacher_motion[offset+local:offset+local+size].copy_(decoded)
            self.info.update(mode='train_features_indices_codec_motion',resident_bytes=index_bytes+feature_bytes+motion_bytes,
                             codec_chunk_rows=8192)

    def window_codes(self,starts,frames):
        ids=starts.to(self.device,non_blocking=True)[:,None]+torch.arange(frames,device=self.device)
        return self.indices[ids].long()

    def reference_codes(self,starts,valid,frames):
        ids=torch.from_numpy(starts).to(self.device)[:,:,None]+torch.arange(frames,device=self.device)
        codes=self.indices[ids].long()
        return codes.masked_fill(~valid.to(self.device)[...,None,None],0)


class ResidentWindowLoader:
    def __init__(self,dataset,sampler,resident,*,batch_size=8):
        self.dataset,self.sampler,self.resident,self.batch_size=dataset,sampler,resident,batch_size
        self.generator=None
        self.frame_starts=np.array([s['frame_start'] for s in dataset.segments],np.int64)

    def __len__(self):return len(self.sampler)//self.batch_size

    def __iter__(self):
        selected=[]
        for index in self.sampler:
            selected.append(index)
            if len(selected)!=self.batch_size:continue
            starts=self.dataset.sample_starts[selected]
            segments=self.dataset.segment_of_start[selected]
            local=self.resident.segment_offsets[segments]+starts-self.frame_starts[segments]
            if (self.resident.segment_offsets[segments]<0).any():raise ValueError('Resident loader cannot read val/test features')
            ids=torch.from_numpy(local).to(self.resident.device)[:,None]+torch.arange(self.dataset.seq_len,device=self.resident.device)
            start_tensor=torch.from_numpy(np.array(starts,copy=True))
            batch=dict(x=self.resident.features[ids],label=torch.from_numpy(self.dataset.labels[starts].astype(np.int64)),
                start=start_tensor,segment_id=torch.from_numpy(np.array(segments,copy=True)),
                vq_indices=self.resident.window_codes(start_tensor,self.dataset.seq_len))
            if self.resident.teacher_motion is not None:batch['teacher_motion']=self.resident.teacher_motion[ids]
            yield batch
            selected=[]
