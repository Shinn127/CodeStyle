import numpy as np
import pytest
from Utils import quat
from Visualization.pose_postprocess import PosePostprocessor,two_bone_target,rotation_between


def rig(**kwargs):
    names=['Hips']+[side+part for side in ('Left','Right') for part in ('UpLeg','Leg','Foot','ToeBase')]
    parents=np.array([-1,0,1,2,3,0,5,6,7])
    offsets=np.array([[0,0,0],[.1,0,0],[0,-.45,0],[0,-.45,0],[0,-.08,.15],
        [-.1,0,0],[0,-.45,0],[0,-.45,0],[0,-.08,.15]],np.float32)
    processor=PosePostprocessor(names,parents,offsets,**kwargs)
    rotations=np.tile(np.array([1.,0,0,0],np.float32),(9,1))
    positions,rotations=processor.fk(rotations,np.array([0,1,0],np.float32))
    return processor,positions,rotations


@pytest.mark.parametrize('target',[[.2,-.75,.15],[3.,-2.,0.],[0,0,0]])
def test_two_bone_ik_is_finite_and_preserves_lengths(target):
    hip=np.array([0,0,0],np.float32);knee=np.array([0,-.4,.1],np.float32);foot=knee*2
    b,c=two_bone_target(hip,knee,foot,np.array(target,np.float32))
    np.testing.assert_allclose(np.linalg.norm(b-hip),np.linalg.norm(knee-hip),atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(c-b),np.linalg.norm(foot-knee),atol=1e-6)
    assert np.isfinite(b).all() and np.isfinite(c).all()


def test_antiparallel_rotation_and_quaternion_sign_are_safe():
    a=np.array([0,1,0],np.float32)
    q=rotation_between(a,-a)
    np.testing.assert_allclose(quat.mul_vec(q,a),-a,atol=1e-6)
    processor,p,r=rig(foot_ik=False)
    _,previous=processor.process(p,r)
    _,current=processor.process(p,-r)
    np.testing.assert_allclose(current,previous,atol=1e-6)


def test_straight_leg_uses_forward_knee_pole():
    hip=np.array([0,0,0],np.float32);knee=np.array([0,-.4,0],np.float32);foot=knee*2
    target=np.array([0,-.7,0],np.float32)
    new_knee,_=two_bone_target(hip,knee,foot,target)
    assert new_knee[2]>0 and abs(new_knee[0])<1e-6


def test_filter_reduces_jitter_keeps_bone_lengths_and_does_not_mutate_input():
    processor,p,r=rig(cutoff=4,foot_ik=False)
    raw=[];filtered=[]
    for i in range(40):
        noisy=p.copy();noisy[:,0]+=.01*(-1 if i%2 else 1)
        before=noisy.copy();out,rot=processor.process(noisy,r)
        np.testing.assert_array_equal(noisy,before)
        np.testing.assert_allclose(np.linalg.norm(rot,axis=-1),1,atol=1e-6)
        for joint in range(1,len(p)):
            np.testing.assert_allclose(np.linalg.norm(out[joint]-out[processor.parents[joint]]),
                np.linalg.norm(processor.offsets[joint]),atol=1e-6)
        raw.append(noisy[0,0]);filtered.append(out[0,0])
    assert np.std(np.diff(filtered[10:]))<np.std(np.diff(raw[10:]))*.5


def test_stance_foot_lock_reduces_slide_and_releases_on_jump():
    processor,p,r=rig(cutoff=0)
    raw=[];fixed=[]
    for i in range(40):
        frame=p.copy();frame[:,0]+=i*.001
        out,_=processor.process(frame,r)
        raw.append(frame[4].copy());fixed.append(out[4].copy())
    raw_slide=np.linalg.norm(np.diff(raw[10:],axis=0)[:,[0,2]],axis=-1).mean()
    fixed_slide=np.linalg.norm(np.diff(fixed[10:],axis=0)[:,[0,2]],axis=-1).mean()
    assert processor.diagnostics['contacts']==2
    assert fixed_slide<raw_slide*.1
    assert abs(processor.pelvis)<=.08
    np.testing.assert_allclose(np.array(fixed)[10:,1],.02,atol=2e-4)
    jumped=p.copy();jumped[:,1]+=.35
    for _ in range(10):out,_=processor.process(jumped,r)
    assert processor.diagnostics['contacts']==0
    assert all(s['weight']==0 for s in processor.contacts)
    assert out[4,1]>.3


def test_noncontact_penetration_raises_foot_without_stretching_leg():
    processor,p,r=rig(cutoff=0)
    p[:,1]-=.03
    for _ in range(12):out,_=processor.process(p,r)
    assert out[4,1]>=.02-1e-5 and out[8,1]>=.02-1e-5
    for joint in (2,3,6,7):
        np.testing.assert_allclose(np.linalg.norm(out[joint]-out[processor.parents[joint]]),.45,atol=1e-6)


def test_fast_swing_and_large_turn_pull_do_not_keep_feet_locked():
    processor,p,r=rig(cutoff=0)
    for _ in range(10):processor.process(p,r)
    moved=p.copy();moved[:,0]+=.3
    processor.process(moved,r)
    assert processor.diagnostics['contacts']==0
    processor.reset()
    for frame in range(20):
        moved=p.copy();moved[:,0]+=frame*.03
        processor.process(moved,r)
        assert processor.diagnostics['contacts']==0


def test_reset_clears_contacts_and_filter_history():
    processor,p,r=rig()
    for _ in range(10):processor.process(p,r)
    assert processor.diagnostics['contacts']==2
    processor.reset()
    assert processor.local is None and all(s['anchor'] is None for s in processor.contacts)
    with pytest.raises(ValueError,match='Nonfinite'):processor.process(p*np.nan,r)
