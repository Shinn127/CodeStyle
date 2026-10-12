"""Live keyboard-driven label controller. GT is used only for initialization."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Utils.interactive_control import LabelControlSession
from Utils.model_loading import build_vqvae_from_checkpoint
from Utils.realtime import parse_style_selector
from Utils.runtime import load_checkpoint,load_norm_stats,resolve_device,configure_torch_runtime,set_seed
from Utils.reproducibility import file_sha256,state_fingerprint
from dataset import norm_stats_fingerprint
from evaluation.simulate_realtime_flowmatch_control import _resolve_base_ckpt_path,_resolve_norm_stats_path,_build_flow_controller


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,default=ROOT/'database_100sty_375_memmap_nomirror')
    p.add_argument('--flow-ckpt',type=Path,default=ROOT/'checkpoints/flowmatch_label_nomirror_150ep/best.pt')
    p.add_argument('--vqvae-ckpt',type=Path,default=None)
    p.add_argument('--norm-stats',type=Path,default=None)
    p.add_argument('--device',default='cuda',choices=('cuda','cpu','auto'))
    p.add_argument('--style',default=None)
    p.add_argument('--source-style',default=None)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--ode-steps',type=int,default=None,
        help='Euler steps per frame; default inherits solver_steps from the checkpoint args (4 for ODE4 runs)')
    p.add_argument('--noise-std',type=float,default=1.,
        help='Per-frame sampling noise scale. The paper protocol uses 1.0 (maximum diversity, '
             'also the noisiest display); 0.25-0.5 keeps near-VQ display smoothness on the current '
             'checkpoint at the cost of deviating from that protocol')
    p.add_argument('--style-blend',type=float,default=.25)
    p.add_argument('--requantize',action=argparse.BooleanOptionalAction,default=True,
        help='Snap the feedback latent to the VQ codebook each step (default, matches formal evaluation); disable for the continuous AE feedback path')
    p.add_argument('--speed',type=float,default=1.2)
    p.add_argument('--turn-speed',type=float,default=90.)
    p.add_argument('--smooth-plan',action=argparse.BooleanOptionalAction,default=False,
        help='Smooth keyboard direction/turn before projecting the future plan; explicit interactive policy')
    p.add_argument('--postprocess',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--foot-ik',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--smooth-cutoff',type=float,default=6.,help='Causal render smoothing cutoff Hz; 0 disables filtering')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--demo',action='store_true',help='Scripted control/style changes for bounded checks')
    p.add_argument('--max-frames',type=int,default=0,help='0 keeps the interactive window open')
    p.add_argument('--screenshot',type=Path,default=None)
    p.add_argument('--report-json',type=Path,default=None)
    args=p.parse_args(argv)
    if args.max_frames<0 or (args.ode_steps is not None and args.ode_steps<1) or not np.isfinite([args.speed,args.turn_speed,args.style_blend,args.smooth_cutoff]).all():p.error('Invalid control settings')
    if not 0<=args.speed<=3 or args.turn_speed<=0 or args.style_blend<0 or args.smooth_cutoff<0:p.error('Invalid speed/transition settings')
    if args.headless and not 1<=args.max_frames<=600:p.error('Headless checks require --max-frames in 1..600')
    if args.headless and args.screenshot:p.error('Screenshots require the GUI')
    return args


def load_session(args):
    device=resolve_device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; select --device cpu')
    configure_torch_runtime(device=device,tf32=False,cudnn_benchmark=False,matmul_precision='highest')
    torch.set_num_threads(2);set_seed(args.seed)
    if not args.flow_ckpt.is_file():
        raise FileNotFoundError(
            f'Flow controller checkpoint not found: {args.flow_ckpt}\n'
            'Old controller checkpoints were removed by the 2026-10-12 cleanup; train one '
            '(e.g. bash scripts/run_train_flowmatch_history_obsnoise_tf100.sh) and pass --flow-ckpt <out>/best.pt.')
    checkpoint=load_checkpoint(args.flow_ckpt,map_location='cpu');config=checkpoint.get('args',{})
    if config.get('conditioning_mode','label')!='label':raise ValueError('Use a label-model checkpoint for this entry')
    kind,base_path=_resolve_base_ckpt_path(args.vqvae_ckpt,None,flow_args=config,flow_ckpt_path=args.flow_ckpt)
    if kind!='vqvae':raise ValueError('This live entry supports the frozen VQ label model')
    norm_path=_resolve_norm_stats_path(args.norm_stats,flow_args=config,flow_ckpt_path=args.flow_ckpt,base_ckpt_path=base_path)
    mean,std=load_norm_stats(norm_path)
    fingerprints=config.get('fingerprints',{})
    if fingerprints.get('norm_fingerprint') and fingerprints['norm_fingerprint']!=norm_stats_fingerprint(mean,std):
        raise ValueError('Normalization differs from the controller manifold')
    names=json.loads((args.db/'StyleNames.json').read_text())
    meta=json.loads((args.db/'meta.json').read_text());segments=json.loads((args.db/'segments.json').read_text())
    if fingerprints.get('split_sha256') and fingerprints['split_sha256']!=file_sha256(args.db/'segments.json'):
        raise ValueError('Database segments differ from the controller training data')
    if any(s.get('style_name',names[s['label']])!=names[s['label']] for s in segments):raise ValueError('Database style label order differs')
    source=parse_style_selector(args.source_style,names)
    if source is None:source=names.index('Neutral') if 'Neutral' in names else 0
    candidates=[s for s in segments if s['label']==source and s['split']=='train' and not s.get('mirror',False)]
    if not candidates:raise ValueError('No unmirrored train initialization clip for source style')
    initial_segment=min(candidates,key=lambda s:({'ID':0,'FW':1}.get(s['suffix'],2),s['frame_start']))
    start=initial_segment['frame_start']+min(60,initial_segment['num_frames']-1)
    data=np.memmap(args.db/'X.f32',mode='r',dtype=np.float32,shape=(int(meta['num_frames']),375))
    initial=np.array(data[start],copy=True);del data
    vq=build_vqvae_from_checkpoint(base_path).to(device)
    model=_build_flow_controller(flow_args=config,flow_ckpt=checkpoint,base_model=vq,base_model_type=kind,
        num_styles=int(config.get('num_styles',len(names))),device=device)
    selected=parse_style_selector(args.style,names)
    extraction=meta.get('generation',{}).get('feature_extraction',meta)
    ode_steps=int(config.get('solver_steps',8)) if args.ode_steps is None else int(args.ode_steps)
    session=LabelControlSession(model,initial,mean,std,names,style=source if selected is None else selected,
        seed=args.seed,steps=ode_steps,noise_std=args.noise_std,style_blend=args.style_blend,requantize=args.requantize,
        window=int(extraction.get('window',60)),stride=int(extraction.get('context_stride',10)),turn_speed=args.turn_speed,
        smooth_plan=getattr(args,'smooth_plan',False))
    session.coordinate_convert=bool(extraction.get('coordinate_convert',False))
    info=dict(checkpoint=str(args.flow_ckpt.resolve()),normalization=str(norm_path),source_entry_id=initial_segment['source_entry_id'],
        initial_frame=int(start),source_style=names[source],steps=ode_steps,noise_std=float(args.noise_std),seed=args.seed,
        fps=60,fp32=True,tf32=False,feedback='clip_quantize_decode' if args.requantize else 'clip_continuous_decode',root='predicted_velocity_integrated',styles=len(names),
        history_conditioning=model.history_conditioning,trajectory_policy='smoothed_commands' if getattr(args,'smooth_plan',False) else 'legacy')
    return session,info


def keyboard_controls(gv,enabled=True):
    if not enabled:return 0.,0.,0.,False
    down=gv.IsKeyDown
    # With Geno facing +Z, +X / positive yaw are character-left.
    return (float(down(gv.KEY_W))-float(down(gv.KEY_S)),float(down(gv.KEY_A))-float(down(gv.KEY_D)),
        float(down(gv.KEY_Q))-float(down(gv.KEY_E)),bool(down(gv.KEY_LEFT_SHIFT) or down(gv.KEY_RIGHT_SHIFT)))


def processing_status(postprocess,foot_ik,cutoff,contacts=0):
    return dict(postprocess=bool(postprocess),smoothing=bool(postprocess and cutoff>0),
        foot_ik=bool(postprocess and foot_ik),bone_fk=bool(postprocess),
        contacts=int(contacts) if postprocess and foot_ik else 0)


def status_button(gv,rect,label,active,*,enabled=True):
    base,border,text,hover=(0xE4F4ECFF,0x36A17BFF,0x1D674DFF,0xCEF0E0FF) if active else (0xF5E9E8FF,0xBF928DFF,0x8B3E35FF,0xECD1CEFF)
    colors={gv.BASE_COLOR_NORMAL:base,gv.BORDER_COLOR_NORMAL:border,gv.TEXT_COLOR_NORMAL:text,
        gv.BASE_COLOR_FOCUSED:hover,gv.TEXT_COLOR_FOCUSED:text,
        gv.BASE_COLOR_DISABLED:0xEAEFF2FF,gv.TEXT_COLOR_DISABLED:0x677886FF}
    previous={key:gv.GuiGetStyle(gv.BUTTON,key) for key in colors}
    try:
        for key,value in colors.items():gv.GuiSetStyle(gv.BUTTON,key,value)
        if not enabled:gv.GuiDisable()
        return bool(gv.GuiButton(rect,label.encode()))
    finally:
        if not enabled:gv.GuiEnable()
        for key,value in previous.items():gv.GuiSetStyle(gv.BUTTON,key,value)


def demo_controls(session):
    frame=session.frames
    if frame in (40,80):session.set_style((session.style+1)%len(session.names))
    return ((1.,0.,0.) if frame<30 else (1.,0.,1.) if frame<60 else (0.,1.,0.) if frame<90 else (-1.,0.,-1.))


def run_headless(session,args):
    latencies=[]
    for frame in range(args.max_frames):
        control=demo_controls(session) if args.demo else (0.,0.,0.)
        started=time.perf_counter();session.step(*control,speed=args.speed);latencies.append((time.perf_counter()-started)*1000)
        if frame%20==0:
            with torch.no_grad():torch.testing.assert_close(session.motion,session.model.vqvae.decode(session.latent),atol=0,rtol=0)
    return latencies


def run_viewer(session,args):
    from Visualization.pose_postprocess import PosePostprocessor
    from Visualization.backends.genoview_backend import _load_genoview_module,GENOVIEW_RESOURCES_DIR
    from Visualization.io.genoview_bvh import _load_template,_template_local_transforms,_expand_selected_to_full,_convert_to_geno_basis
    gv=_load_genoview_module()
    template=_load_template(GENOVIEW_RESOURCES_DIR/'Geno_bind.bvh')
    parents=np.asarray(template['parents'],np.int32);local_pos,local_rot=_template_local_transforms(template,0)
    processor=PosePostprocessor(template['names'],parents,local_pos,cutoff=args.smooth_cutoff,foot_ik=args.foot_ik)
    postprocess=args.postprocess;pose_frame=None;pose_cost=[];camera_target=None
    gv.SetConfigFlags(gv.FLAG_MSAA_4X_HINT|gv.FLAG_VSYNC_HINT)
    gv.InitWindow(1280,800,b'Live label flow controller | WASD + QE')
    model=None;shader=None;latencies=[]
    try:
        if not gv.IsWindowReady():raise RuntimeError('Could not open a raylib window')
        gv.SetTargetFPS(60)
        fragment=b'''#version 300 es
precision highp float;
in vec3 fragNormal; in vec3 fragPosition; in vec4 fragColor; uniform vec4 colDiffuse; uniform vec3 cameraPosition;
out vec4 finalColor;
void main(){vec3 n=normalize(fragNormal);float light=0.58+0.42*max(0.0,dot(n,normalize(vec3(0.5,0.9,0.7))));
float rim=pow(1.0-max(0.0,dot(n,normalize(cameraPosition-fragPosition))),3.0);
finalColor=vec4(colDiffuse.rgb*fragColor.rgb*light+vec3(0.05,0.08,0.12)*rim,1.0);}
'''
        shader=gv.LoadShaderFromMemory((GENOVIEW_RESOURCES_DIR/'skinnedBasic.vs').read_bytes(),fragment)
        if shader.id==gv.rlGetShaderIdDefault():raise RuntimeError('Could not compile the skinning shader')
        model=gv.LoadGenoModel(str(GENOVIEW_RESOURCES_DIR/'Geno.bin').encode())
        if model.boneCount!=len(parents):raise ValueError('Geno mesh/template joint orders differ')
        model.materials[0].shader=shader
        camera_location=gv.GetShaderLocation(shader,b'cameraPosition')
        camera_value=gv.ffi.new('float[3]')
        bind_pos,bind_rot=gv.GetModelBindPoseAsNumpyArrays(model)
        camera=gv.Camera();camera.distance=5.2;camera.altitude=.28;camera.azimuth=np.pi
        search=gv.ffi.new('char[128]');scroll=gv.ffi.new('int *',max(0,session.style-4));active=gv.ffi.new('int *')
        speed=gv.ffi.new('float *',args.speed)
        editing=False;paused=False;skeleton=False;trajectory=True;last_query='';pending_toggle=None
        accumulator=0.;previous=time.perf_counter();render_frames=0;infer_ms=0.
        gv.GuiSetStyle(gv.DEFAULT,gv.TEXT_SIZE,16)
        while not gv.WindowShouldClose() and (args.max_frames==0 or render_frames<args.max_frames):
            now=time.perf_counter();wall_dt=min(now-previous,.1);previous=now
            if pending_toggle=='postprocess':postprocess=not postprocess;processor.reset();pose_frame=None
            elif pending_toggle=='ik' and postprocess:processor.foot_ik=not processor.foot_ik;processor.reset();pose_frame=None
            pending_toggle=None
            mouse=gv.GetMousePosition();panel=mouse.x>=968
            search_rect=gv.Rectangle(994,318,258,30)
            if gv.IsMouseButtonPressed(gv.MOUSE_BUTTON_LEFT) and not gv.CheckCollisionPointRec(mouse,search_rect):editing=False
            enabled=not editing and gv.IsWindowFocused()
            if enabled:
                if gv.IsKeyPressed(gv.KEY_SPACE):paused=not paused;accumulator=0.
                if gv.IsKeyPressed(gv.KEY_R):session.reset();processor.reset();pose_frame=None;camera_target=None;accumulator=0.
                if gv.IsKeyPressed(gv.KEY_LEFT_BRACKET):session.set_style((session.style-1)%len(session.names))
                if gv.IsKeyPressed(gv.KEY_RIGHT_BRACKET):session.set_style((session.style+1)%len(session.names))
                if gv.IsKeyPressed(gv.KEY_MINUS):speed[0]=max(0.,speed[0]-.2)
                if gv.IsKeyPressed(gv.KEY_EQUAL):speed[0]=min(3.,speed[0]+.2)
                if gv.IsKeyPressed(gv.KEY_F1):skeleton=not skeleton
                if gv.IsKeyPressed(gv.KEY_T):trajectory=not trajectory
                if gv.IsKeyPressed(gv.KEY_P):postprocess=not postprocess;processor.reset();pose_frame=None
                if gv.IsKeyPressed(gv.KEY_I) and postprocess:processor.foot_ik=not processor.foot_ik;processor.reset();pose_frame=None
            forward,side,turn,sprint=keyboard_controls(gv,enabled)
            factor=-1 if session.coordinate_convert else 1
            accumulator=0. if paused else min(accumulator+(session.planner.dt if args.demo else wall_dt),3*session.planner.dt)
            for _ in range(2):
                if accumulator<session.planner.dt:break
                controls=demo_controls(session) if args.demo else (forward,side*factor,turn*factor)
                started=time.perf_counter();session.step(*controls,speed=min(3.,float(speed[0])*(1.7 if sprint else 1.)))
                infer_ms=(time.perf_counter()-started)*1000;latencies.append(infer_ms);accumulator-=session.planner.dt
            if pose_frame!=session.frames:
                selected_pos,selected_rot=session.selected_pose()
                selected_pos,selected_rot=_convert_to_geno_basis(selected_pos,selected_rot,enabled=session.coordinate_convert)
                full_pos,full_rot=_expand_selected_to_full(selected_pos,selected_rot,full_names=template['names'],full_parents=parents,
                    template_local_pos=local_pos,template_local_rot=local_rot)
                if postprocess:
                    started=time.perf_counter()
                    dt=(1 if pose_frame is None else session.frames-pose_frame)*session.planner.dt
                    full_pos[0],full_rot[0]=processor.process(full_pos[0],full_rot[0],dt=dt)
                    pose_cost.append((time.perf_counter()-started)*1000)
                if not np.isfinite(full_pos).all() or not np.isfinite(full_rot).all():raise ValueError('Nonfinite rendered pose')
                pose_frame=session.frames
            gv.UpdateModelPoseFromNumpyArrays(model,bind_pos,bind_rot,full_pos[0],full_rot[0])
            root=np.array([factor*session.planner.position[0],0.,session.planner.position[1]])
            delta=gv.GetMouseDelta();orbit=not panel and gv.IsMouseButtonDown(gv.MOUSE_BUTTON_RIGHT)
            follow=root+np.array([0,1.,0])
            if camera_target is None or not postprocess:camera_target=follow.copy()
            else:camera_target+=(1-np.exp(-2*np.pi*6*max(wall_dt,1./120)))*(follow-camera_target)
            camera.update(gv.Vector3(*camera_target),delta.x if orbit else 0.,delta.y if orbit else 0.,0.,0.,
                gv.GetMouseWheelMove() if not panel else 0.,max(wall_dt,1./120))
            eye=camera.cam3d.position;camera_value[0]=eye.x;camera_value[1]=eye.y;camera_value[2]=eye.z
            gv.SetShaderValue(shader,camera_location,camera_value,gv.SHADER_UNIFORM_VEC3)
            gv.BeginDrawing();gv.ClearBackground(gv.Color(232,238,242,255));gv.BeginMode3D(camera.cam3d)
            gv.DrawPlane(gv.Vector3(*root),gv.Vector2(100,100),gv.Color(216,224,230,255))
            gv.rlPushMatrix();gv.rlTranslatef(float(np.floor(root[0])),.002,float(np.floor(root[2])))
            gv.DrawGrid(32,1.);gv.rlPopMatrix()
            for _,_,foot,toe in processor.legs:
                point=full_pos[0,foot];alpha=int(35*max(0.,1-max(0.,float(point[1])-.08)/.4))
                gv.DrawCylinder(gv.Vector3(point[0],.006,point[2]),.12,.12,.002,20,gv.Color(48,61,72,alpha))
            gv.DrawModel(model,gv.Vector3(0,0,0),1.,gv.Color(64,124,196,255))
            if skeleton:gv.DrawSkeleton(full_pos[0],full_rot[0],parents,gv.Color(240,138,45,255))
            if trajectory:
                p=session.debug['positions'].copy();d=session.debug['directions'].copy();p[:,0]*=factor;d[:,0]*=factor
                for i in range(len(p)):
                    color=gv.Color(38,164,119,255) if session.planner.offsets[i]>=0 else gv.Color(130,145,160,255)
                    point=np.array([p[i,0],.04,p[i,1]])
                    if i:gv.DrawLine3D(gv.Vector3(p[i-1,0],.04,p[i-1,1]),gv.Vector3(*point),color)
                    gv.DrawSphere(gv.Vector3(*point),.035,color)
                    gv.DrawLine3D(gv.Vector3(*point),gv.Vector3(*(point+np.array([d[i,0],0,d[i,1]])*.22)),color)
            facing=planar_facing(session.planner.yaw,factor)
            gv.DrawLine3D(gv.Vector3(*(root+np.array([0,.03,0]))),gv.Vector3(*(root+facing*.8+np.array([0,.03,0]))),gv.Color(239,153,48,255))
            gv.EndMode3D()
            gv.DrawRectangle(968,0,312,800,gv.Color(247,249,251,250))
            text=lambda message,y,size=17,color=None:gv.DrawText(message.encode(),994,y,size,color or gv.Color(40,57,74,255))
            text('LABEL CONTROL',26,24)
            model_name=args.flow_ckpt.parent.name.removeprefix('flowmatch_label_')
            text(f'{model_name[:25]} | 60 Hz',63,13)
            text(f'History: {2 if session.model.history_conditioning else 1} frame(s) | Noise: {session.noise_std:g}',82,13)
            text(f'{session.style:02d}  {session.names[session.style]}',100,19)
            text(f'{gv.GetFPS()} render FPS  |  {infer_ms:.1f} ms inference',133,15)
            text(f'Frames {session.frames} | {"PAUSED" if paused else "RUNNING"}',157,15)
            status=processing_status(postprocess,processor.foot_ik,processor.cutoff,processor.diagnostics['contacts'])
            on_color=gv.Color(28,155,99,255);off_color=gv.Color(166,78,70,255)
            if status_button(gv,gv.Rectangle(994,184,258,30),f'Post-processing: {"ON" if status["postprocess"] else "OFF"} [P]',status['postprocess']):
                pending_toggle='postprocess'
            gv.DrawCircle(1007,199,4,on_color if status['postprocess'] else off_color)
            ik_text='ON [I]' if status['foot_ik'] else 'OFF [I]' if postprocess else 'INACTIVE'
            if status_button(gv,gv.Rectangle(994,222,258,30),f'Foot IK: {ik_text}',status['foot_ik'],enabled=postprocess):pending_toggle='ik'
            gv.DrawCircle(1007,237,4,on_color if status['foot_ik'] else off_color)
            smooth=f'ON ({processor.cutoff:g} Hz base)' if status['smoothing'] else 'OFF'
            text(f'Pose smooth: {smooth}',264,14)
            if postprocess:text(f'IK contact {status["contacts"]}/2 | full blend {processor.diagnostics.get("fully_locked",0)}/2',286,13)
            else:text('RAW pose | IK inactive (master OFF)',286,13)
            text('Search style:',300,14)
            if gv.GuiTextBox(search_rect,search,128,editing):editing=not editing
            query=gv.ffi.string(search).decode(errors='replace').lower()
            if query!=last_query:scroll[0]=0;last_query=query
            filtered=[i for i,name in enumerate(session.names) if query in name.lower()]
            active[0]=filtered.index(session.style) if session.style in filtered else -1
            if filtered:
                before=active[0]
                gv.GuiListView(gv.Rectangle(994,358,258,188),(';'.join(f'{i:02d} {session.names[i]}' for i in filtered)).encode(),scroll,active)
                if active[0]!=before and 0<=active[0]<len(filtered):session.set_style(filtered[active[0]])
            else:text('No matching style',368,16)
            text(f'Walk speed  {speed[0]:.1f} m/s',565,16)
            gv.GuiSlider(gv.Rectangle(994,596,258,18),b'',b'',speed,0.,3.)
            for i,line in enumerate(['W/S forward/back | A/D left/right','Q turn left | E turn right','Shift sprint | -/+ speed | [/] style','Space pause | R reset','P processing | I foot IK (or click)','Right-drag orbit | Wheel zoom','T path | F1 bones | FP32',f'ESC close | Euler{session.steps} | {"RQ" if session.requantize else "AE"} feedback']):text(line,632+i*20,13)
            render_frames+=1
            if args.screenshot and args.max_frames and render_frames==args.max_frames:
                args.screenshot.parent.mkdir(parents=True,exist_ok=True)
                gv.rlDrawRenderBatchActive()
                image=gv.LoadImageFromScreen()
                try:
                    if not gv.ExportImage(image,str(args.screenshot.resolve()).encode()):raise RuntimeError('Screenshot export failed')
                finally:gv.UnloadImage(image)
            gv.EndDrawing()
    finally:
        if model is not None:gv.UnloadModel(model)
        if shader is not None:gv.UnloadShader(shader)
        gv.CloseWindow()
    effective=processing_status(postprocess,processor.foot_ik,processor.cutoff,processor.diagnostics['contacts'])
    session.render_diagnostics=dict(postprocess=postprocess,foot_ik=processor.foot_ik,foot_ik_active=effective['foot_ik'],
        smoothing_active=effective['smoothing'],smooth_cutoff=args.smooth_cutoff,
        postprocess_ms_mean=float(np.mean(pose_cost)) if pose_cost else 0.,**processor.diagnostics)
    return latencies


def planar_facing(yaw,factor=1):
    return np.array([factor*np.sin(yaw),0.,np.cos(yaw)],np.float32)


def main(argv=None):
    args=parse_args(argv)
    for name in ('db','flow_ckpt','vqvae_ckpt','norm_stats'):
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.expanduser().resolve())
    session,info=load_session(args)
    print('[Live]',json.dumps(info,ensure_ascii=False),flush=True)
    before=state_fingerprint(session.model.vqvae.state_dict())
    latencies=run_headless(session,args) if args.headless else run_viewer(session,args)
    after=state_fingerprint(session.model.vqvae.state_dict())
    if before!=after:raise AssertionError('Interactive inference modified the frozen VQ')
    measured=latencies[min(10,len(latencies)-1):] if latencies else []
    info.update(frames=session.frames,final_style=session.names[session.style],vq_unchanged=True,
        inference_ms_mean=float(np.mean(measured)) if measured else None,
        inference_ms_p95=float(np.percentile(measured,95)) if measured else None,
        root_position=session.planner.position.tolist(),root_yaw_degrees=float(np.degrees(session.planner.yaw)))
    if hasattr(session,'render_diagnostics'):info['render']=session.render_diagnostics
    if args.report_json:
        args.report_json.parent.mkdir(parents=True,exist_ok=True);args.report_json.write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print('[Completed]',json.dumps(info,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
