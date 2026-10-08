use nalgebra::{Matrix3, Vector3};
use rusty_pall::{
    rig_motion::{project_rotation, rotation6, topology},
    usd_rig::parents,
};

#[test]
fn root_velocity_uses_previous_velocity_and_destination_facing() {
    let mut values = vec![0f32; 24];
    values[2] = 2.;
    values[3] = 3.; // absolute height, two frames
    values[6] = 1.;
    values[14] = 1.; // frame zero identity
    values[11] = -1.;
    values[15] = 1.; // frame one +90 degrees around Y
    values[18] = 1.; // frame zero x velocity
    let positions = rusty_pall::rig_motion::decode_root(&values, 2).unwrap();
    assert!((positions[0] - Vector3::new(0., 2., 0.)).norm() < 1e-10);
    assert!((positions[1] - Vector3::new(0., 3., 1.)).norm() < 1e-10);
}

#[test]
fn rejects_duplicate_or_unsorted_joint_hierarchies() {
    assert!(parents(&["root".into(), "root".into()]).is_err());
    assert!(parents(&["root".into(), "root/missing/child".into()]).is_err());
    assert!(parents(&["root".into(), "other".into()]).is_err());
    assert_eq!(
        parents(&["root".into(), "root/a".into(), "root/a/b".into()]).unwrap(),
        vec![None, Some(0), Some(1)]
    );
}

#[test]
fn graph_relations_and_normalized_laplacian_match_tree_definition() {
    let p = vec![None, Some(0), Some(0), Some(1)];
    let (dist, rel, depth, spectral) = topology(&p, 6, 4, 8);
    assert_eq!(&depth[..4], &[0, 1, 1, 2]);
    assert_eq!(rel[1], 2); // root -> child
    assert_eq!(rel[6], 1); // child -> root
    assert_eq!(rel[6 + 2], 3); // siblings
    assert_eq!(rel[2 * 6 + 2], 5); // leaf diagonal
    assert_eq!(dist[2 * 6 + 3], 3);
    assert_eq!(&dist[4 * 6..], &[0; 12]);
    // Nontrivial eigenvectors are orthogonal to sqrt(degree), the null mode
    // of the symmetric normalized Laplacian. Small skeletons pad frequencies.
    let degree = [2f64, 2., 1., 1.];
    for k in 0..3 {
        let dot: f64 = (0..4)
            .map(|i| spectral[i * 4 + k] as f64 * degree[i].sqrt())
            .sum();
        assert!(dot.abs() < 1e-5);
        let norm: f32 = (0..4).map(|i| spectral[i * 4 + k].powi(2)).sum();
        assert!((norm - 1.).abs() < 1e-5);
    }
    for i in 0..6 {
        assert_eq!(spectral[i * 4 + 3], 0.);
    }
}

#[test]
fn six_dimensional_rotation_uses_columns_and_rejects_degeneracy() {
    assert_eq!(
        rotation6([1., 0., 0., 0., 1., 0.]).unwrap(),
        Matrix3::identity()
    );
    let r = rotation6([0., 1., 0., -1., 0., 0.]).unwrap();
    assert!((r * Vector3::x() - Vector3::y()).norm() < 1e-10);
    assert!(rotation6([0.; 6]).is_err());
    assert!(rotation6([1., 0., 0., 2., 0., 0.]).is_err());
    let mean = project_rotation(Matrix3::identity() + r);
    assert!((mean.transpose() * mean - Matrix3::identity()).norm() < 1e-10);
    assert!((mean.determinant() - 1.).abs() < 1e-10);
}

#[test]
fn usd_rest_pose_round_trip_preserves_rotated_joints_under_scaled_parent() -> anyhow::Result<()> {
    use rusty_pall::{
        rig_motion::{Prepared, validate_export},
        usd_rig::{Rig, position},
    };
    let dir = std::env::temp_dir().join(format!(
        "rusti-usd-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&dir)?;
    let source = dir.join("rig.usda");
    std::fs::write(
        &source,
        r#"#usda 1.0
(
    upAxis = "Y"
)
def Xform "World" {
    double3 xformOp:translate = (4,5,6)
    double3 xformOp:scale = (2,2,2)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    def SkelRoot "Rig" {
        def Skeleton "Skeleton" {
            uniform token[] joints = ["root", "root/a", "root/a/b", "root/c"]
            uniform matrix4d[] restTransforms = [
                ((0,1,0,0),(-1,0,0,0),(0,0,1,0),(0,2,0,1)),
                ((1,0,0,0),(0,0,1,0),(0,-1,0,0),(1,0,0,1)),
                ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,1,0,1)),
                ((1,0,0,0),(0,1,0,0),(0,0,1,0),(-1,0,0,1))
            ]
            uniform matrix4d[] bindTransforms = [
                ((0,1,0,0),(-1,0,0,0),(0,0,1,0),(0,2,0,1)),
                ((0,1,0,0),(0,0,1,0),(1,0,0,0),(0,3,0,1)),
                ((0,1,0,0),(0,0,1,0),(1,0,0,0),(0,3,1,1)),
                ((0,1,0,0),(-1,0,0,0),(0,0,1,0),(0,1,0,1))
            ]
        }
        def Mesh "Mesh" {
            rel skel:skeleton = </World/Rig/Skeleton>
            int[] primvars:skel:jointIndices = [0,1,2,3]
            float[] primvars:skel:jointWeights = [1,1,1,1]
        }
    }
}
"#,
    )?;
    let rig = Rig::open(&source)?;
    assert!((position(&rig.world_rest[0]) - Vector3::new(4., 9., 6.)).norm() < 1e-10);
    let prepared = Prepared {
        input: source.to_string_lossy().into(),
        prompt: "rest".into(),
        labels: vec!["Bone".into(); 4],
        joints: rig.joints.clone(),
        parents: rig.parents.clone(),
        positions: rig.world_rest.iter().map(|m| position(m).into()).collect(),
        canonical_rotation: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        shift: [0.; 3],
        scale: 1.,
        width: 4,
        frames: 2,
        frequencies: 3,
        tpos: vec![],
        parent_features: vec![],
        graph_dist: vec![],
        relations: vec![],
        depths: vec![],
        spectral: vec![],
        caption: vec![],
        joint_embeddings: vec![],
    };
    let mut features = vec![0f32; 4 * 12 * 2];
    for j in 0..4 {
        for t in 0..2 {
            for d in 0..3 {
                features[(j * 12 + d) * 2 + t] = prepared.positions[j][d] as f32;
            }
            features[(j * 12 + 3) * 2 + t] = 1.;
            features[(j * 12 + 7) * 2 + t] = 1.;
        }
    }
    // The root representation has no absolute X/Z; shift restores authored placement.
    let mut prepared = prepared;
    prepared.shift = [-4., 0., -6.];
    let out = dir.join("rest.usda");
    prepared.export(&rig, &features, &out)?;
    validate_export(&rig, &out, 2, true)?;
    // A self round trip missed invalid apiSchemas syntax in openusd 0.7.
    // Use the independent reference parser when the USD tools are installed.
    match std::process::Command::new("usdcat")
        .arg(&out)
        .arg("-o")
        .arg(dir.join("reference.usdc"))
        .output()
    {
        Ok(result) => assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("usdcat unavailable; independent USD parse check skipped")
        }
        Err(e) => return Err(e.into()),
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
fn singleton_api_schema_lists_keep_list_edit_semantics() {
    use rusty_pall::rig_motion::fix_api_schema_lists;
    let source = "    prepend apiSchemas = \"SkelBindingAPI\"\n    apiSchemas = [\"A\", \"B\"]\n    string note = \"SkelBindingAPI\"\n";
    let fixed = fix_api_schema_lists(source);
    assert_eq!(
        fixed,
        "    prepend apiSchemas = [\"SkelBindingAPI\"]\n    apiSchemas = [\"A\", \"B\"]\n    string note = \"SkelBindingAPI\"\n"
    );
    assert_eq!(fix_api_schema_lists(&fixed), fixed);
}
