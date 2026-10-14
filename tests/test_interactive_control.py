"""Keyboard trajectory layout, coordinate signs, and autonomous feedback."""
import json

import numpy as np
import pytest
import torch

from Utils.interactive_control import InteractiveTrajectory,LabelControlSession
from Utils.motion375 import reconstruct_root_track_from_375
from Utils.reproducibility import state_fingerprint
from test_controller_repair import controller


def test_trajectory_layout_past_future_stop_and_diagonal_speed():
    planner=InteractiveTrajectory()
    for _ in range(65):planner.advance(np.array([0.,0.,1.],np.float32))
    trajectory,debug=planner.condition(forward=1,side=1,speed=1.2)
    blocks=trajectory.reshape(7,12);center=6
    np.testing.assert_array_equal(blocks[:2,center],0.)
    np.testing.assert_allclose(blocks[2:4,center],[0,1],atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(blocks[2:4],axis=0),1.,atol=1e-6)
    np.testing.assert_allclose(blocks[6],np.linalg.norm(blocks[4:6],axis=0),atol=1e-7)
    assert blocks[1,0]<-.99 and blocks[0,-1]>0 and blocks[1,-1]>0
    assert blocks[6,-1]<=1.201 and debug['positions'].shape==(12,2)
    stopped,_=planner.condition(speed=0.)
    assert stopped[72+11]<.002


def test_root_integration_matches_existing_motion375_degrees_and_previous_frame():
    planner=InteractiveTrajectory();root=np.array([.3,60.,.8],np.float32)
    features=np.zeros((61,375),np.float32);features[:,84:87]=root
    features[:,159:231]=np.tile([1,0,0],24);features[:,231:303]=np.tile([0,1,0],24)
    positions,rotations=reconstruct_root_track_from_375(features)
    for _ in range(60):planner.advance(root)
    np.testing.assert_allclose(planner.position,positions[-1,[0,2]],atol=2e-6)
    np.testing.assert_allclose(planner.yaw,np.pi/3,atol=1e-6)
    before=planner.position.copy();planner.condition(turn=1)
    np.testing.assert_array_equal(planner.position,before)  # intent never overwrites generated root


def test_idle_start_has_commanded_center_velocity_without_moving_actual_root():
    planner=InteractiveTrajectory()
    for _ in range(40):trajectory,_=planner.condition(forward=1,speed=1.2)
    assert trajectory.reshape(7,12)[5,6]>1.19
    np.testing.assert_array_equal(planner.position,[0,0])
    np.testing.assert_array_equal(planner.velocity,[0,0])
    for _ in range(60):trajectory,_=planner.condition(speed=0)
    assert trajectory.reshape(7,12)[6,6]<.001


@pytest.mark.parametrize('turn',[-1,1])
def test_qe_turn_sign_and_local_heading_invariance(turn):
    planner=InteractiveTrajectory();x,_=planner.condition(forward=1,turn=turn)
    assert np.sign(x[24+11])==turn
    planner.reset();planner.yaw=1.1;planner.desired_yaw=1.1
    planner.history.clear();planner.history.append((planner.position.copy(),planner.yaw,planner.velocity.copy()))
    y,_=planner.condition(forward=1)
    reference=InteractiveTrajectory().condition(forward=1)[0]
    np.testing.assert_allclose(y,reference,atol=1e-6)


def test_label_live_feedback_reset_rng_and_style_change_preserve_history():
    model=controller();names=['a','b','c'][:model.num_styles]
    initial=np.zeros(375,np.float32)
    session=LabelControlSession(model,initial,np.zeros(375,np.float32),np.ones(375,np.float32),names,steps=2,style_blend=.05)
    vq=state_fingerprint(model.vqvae.state_dict())
    first=session.step(forward=1).copy()
    torch.testing.assert_close(session.motion,model.vqvae.decode(session.latent),atol=0,rtol=0)
    previous=session.latent.clone();location=session.planner.position.copy()
    session.set_style(1)
    torch.testing.assert_close(session.latent,previous,atol=0,rtol=0)
    np.testing.assert_array_equal(session.planner.position,location)
    for _ in range(4):session.step(side=1,turn=-1)
    torch.testing.assert_close(session.current_feature,session.feature(1),atol=0,rtol=0)
    assert session.selected_pose()[0].shape==(1,24,3)
    session.set_style(0);session.reset()
    np.testing.assert_array_equal(session.step(forward=1),first)
    assert state_fingerprint(model.vqvae.state_dict())==vq
    assert all(p.grad is None and p.dtype==torch.float32 for p in model.parameters())
    with pytest.raises(ValueError,match='range'):session.set_style(100)


def test_continuous_feedback_skips_codebook_and_matches_decode():
    model=controller();names=['a','b','c'][:model.num_styles]
    initial=np.zeros(375,np.float32)
    session=LabelControlSession(model,initial,np.zeros(375,np.float32),np.ones(375,np.float32),names,
        steps=2,style_blend=0,requantize=False)
    assert session.requantize is False
    vq=state_fingerprint(model.vqvae.state_dict())
    torch.testing.assert_close(session.motion,model.vqvae.decode(session.latent),atol=0,rtol=0)
    for _ in range(3):session.step(forward=1)
    torch.testing.assert_close(session.motion,model.vqvae.decode(session.latent),atol=0,rtol=0)
    assert not torch.equal(session.latent,model.requantize_latent(session.latent))
    assert state_fingerprint(model.vqvae.state_dict())==vq
    with pytest.raises(ValueError,match='requantize'):
        LabelControlSession(model,initial,np.zeros(375,np.float32),np.ones(375,np.float32),names,
            steps=2,style_blend=.05,requantize=1)


def test_keyboard_focus_and_opposing_keys():
    from Visualization.cli.interactive_label_control import keyboard_controls
    class Keys:
        KEY_W='w';KEY_S='s';KEY_D='d';KEY_A='a';KEY_E='e';KEY_Q='q';KEY_LEFT_SHIFT='shift';KEY_RIGHT_SHIFT='right_shift'
        @staticmethod
        def IsKeyDown(key):return key in {'w','d','q','shift'}
    assert keyboard_controls(Keys)==(1.,-1.,1.,True)
    assert keyboard_controls(Keys,False)==(0.,0.,0.,False)


@pytest.mark.parametrize('key,side,turn',[('a',1.,0.),('d',-1.,0.),('q',0.,1.),('e',0.,-1.)])
def test_corrected_geno_left_right_key_signs(key,side,turn):
    from Visualization.cli.interactive_label_control import keyboard_controls
    class Keys:
        KEY_W='w';KEY_S='s';KEY_D='d';KEY_A='a';KEY_E='e';KEY_Q='q';KEY_LEFT_SHIFT='shift';KEY_RIGHT_SHIFT='right_shift'
        @staticmethod
        def IsKeyDown(candidate):return candidate==key
    assert keyboard_controls(Keys)==(0.,side,turn,False)


def test_opposing_direction_keys_cancel():
    from Visualization.cli.interactive_label_control import keyboard_controls
    class Keys:
        KEY_W='w';KEY_S='s';KEY_D='d';KEY_A='a';KEY_E='e';KEY_Q='q';KEY_LEFT_SHIFT='shift';KEY_RIGHT_SHIFT='right_shift'
        @staticmethod
        def IsKeyDown(key):return key in {'w','s','a','d','q','e'}
    assert keyboard_controls(Keys)==(0.,0.,0.,False)


@pytest.mark.parametrize('master,ik,cutoff',[(False,True,6),(False,False,6),(True,True,6),(True,False,6),(True,True,0),(True,False,0)])
def test_ui_reports_effective_processing_state(master,ik,cutoff):
    from Visualization.cli.interactive_label_control import processing_status
    status=processing_status(master,ik,cutoff,contacts=2)
    assert status['postprocess']==master
    assert status['smoothing']==bool(master and cutoff>0)
    assert status['foot_ik']==bool(master and ik)
    assert status['bone_fk']==master
    assert status['contacts']==(2 if master and ik else 0)


def test_reel_schedule_is_bounded_and_swaps_styles_in_order():
    from Visualization.cli.interactive_label_control import REEL_PHASES,REEL_STYLES,reel_controls
    class Session:
        def __init__(self):self.frames=0;self.names=['Neutral','Robot','Old','HighKnees','ArmsAboveHead'];self.style=0;self.swaps=[]
        def set_style(self,style):self.style=int(style);self.swaps.append(style)
    session=Session()
    total=sum(length for length,_ in REEL_PHASES)
    seen=[]
    for frame in range(max(at for at,_ in REEL_STYLES)+30):
        session.frames=frame
        forward,side,turn,speed=reel_controls(session)
        assert -1<=forward<=1 and -1<=side<=1 and -1<=turn<=1 and 0<=speed<=3
        seen.append(session.style)
    assert [name for _,name in REEL_STYLES]==['Robot','Old','HighKnees','ArmsAboveHead','Neutral']
    assert session.swaps==[1,2,3,4,0]
    assert seen[-1]==0 and len(set(seen))==len(REEL_STYLES)
    assert total==sum(length for length,_ in REEL_PHASES)==720


def test_frame_recorder_encodes_and_counts_frames(tmp_path):
    import shutil,subprocess
    if shutil.which('ffmpeg') is None:pytest.skip('ffmpeg unavailable')
    from Visualization.cli.interactive_label_control import FrameRecorder
    path=tmp_path/'clip.mp4'
    recorder=FrameRecorder(path,fps=30)
    width,height=64,48
    checker=np.zeros((height,width,4),np.uint8)
    checker[...,3]=255
    for frame in range(30):
        checker[...,0]=(frame*8)%256
        recorder.write(checker.tobytes(),width,height)
    frames=recorder.close()
    assert frames==30 and path.stat().st_size>0
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames',
        '-of','json',str(path)],capture_output=True,text=True)
    if probe.returncode==0:
        info=json.loads(probe.stdout)['streams'][0]
        assert (info['width'],info['height'])==(64,48)


def test_parse_args_rejects_record_in_headless_mode():
    from Visualization.cli.interactive_label_control import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--headless','--max-frames','10','--record','/tmp/x.mp4'])


def test_hidden_and_reel_flags_reach_the_args():
    from Visualization.cli.interactive_label_control import parse_args
    args=parse_args(['--hidden','--reel','--max-frames','30'])
    assert args.hidden and args.reel and not args.headless
