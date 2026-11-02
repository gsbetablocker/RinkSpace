#pragma once
#include "core_types.hpp"

namespace rinkspace {
// These are finite layout146 identities, not expressions or numerical
// equivalence classes. Keep the legacy relation evaluator as the domain oracle.
constexpr int ice_relation_legacy_count=206;
constexpr int ice_relation_deviation_count=107;
constexpr int ice_relation_independent_count=192;

enum class ConstraintShape { affine,nonlinear,supporting_bundle };
enum class IceCanonicalKind { invalid,relation,deviation_box };
enum class IceRelationBound { none,lower,upper };
enum class IceRelationFamily {
    invalid,row_gap,row_clearance,ring_order,
    lane_house_order,lane_centre_slope,lane_right_after_left,
    lane_transverse_after_left,lane_right_after_transverse,
    lane_centre_end_after_house,lane_centre_end_before_right,
    lane_right_x,lane_transverse_x,lane_side_slope,lane_width_left,
    lane_width_right,lane_left_goal_support,lane_house_goal_support,
    lane_separation,carrier_order,faceoff_goal_order,
    trapezoid_board_support,trapezoid_goal_support
};
// Physical reference categories only. They do not change the legacy margin,
// its strict epsilon inset, active tolerance, or derivative normalization.
enum class IceRelationReferenceScale {
    invalid,direction_unit,nominal_pitch,nominal_row_clearance,
    nominal_ring_outer_radius,nominal_house_run,nominal_lane_width,
    board_half_length,board_half_width,nominal_goal_offset,
    nominal_trapezoid_goal_span,deviation_unit
};

struct IceRelationMeta {
    int legacy_id=-1;
    // Unified identity: canonical relation IDs remain in [0,206); deviation
    // box IDs use 206+2*index+(upper?0:1). IDs are keys, not dense row offsets.
    int canonical_id=-1;
    IceCanonicalKind canonical_kind=IceCanonicalKind::invalid;
    int canonical_index=-1; // canonical legacy relation ID or deviation index
    IceRelationFamily family=IceRelationFamily::invalid;
    int row=-1,column=-1,side=0,index=-1;
    // side is the spatial sign (-1/+1), independent of relative bound.
    // index is the ring, carrier adjacency, faceoff/trapezoid, or sample index.
    IceRelationBound bound=IceRelationBound::none;
    // shape describes the ORIGINAL raw margin in the general scene chart.
    // A log/ordered nominal coordinate multiplied by a deviation is nonlinear.
    ConstraintShape shape=ConstraintShape::nonlinear;
    ConstraintShape canonical_shape=ConstraintShape::nonlinear;
    bool strict=false;
    IceRelationReferenceScale reference_scale=IceRelationReferenceScale::invalid;
    IceRelationReferenceScale canonical_reference_scale=IceRelationReferenceScale::invalid;
    RS_HD constexpr bool valid()const{
        return legacy_id>=0&&legacy_id<ice_relation_legacy_count&&
            family!=IceRelationFamily::invalid&&canonical_kind!=IceCanonicalKind::invalid;
    }
};

RS_HD constexpr int ice_deviation_box_canonical_id(int deviation,IceRelationBound bound){
    return deviation<0||deviation>=ice_relation_deviation_count||
        (bound!=IceRelationBound::lower&&bound!=IceRelationBound::upper)?-1:
        ice_relation_legacy_count+2*deviation+(bound==IceRelationBound::upper?0:1);
}

RS_HD constexpr IceRelationMeta ice_relation_meta(int legacy_id){
    IceRelationMeta m;
    if(legacy_id<0||legacy_id>=ice_relation_legacy_count)return m;
    m.legacy_id=legacy_id;m.canonical_id=legacy_id;
    m.canonical_kind=IceCanonicalKind::relation;m.canonical_index=legacy_id;
    if(legacy_id<20){
        m.column=legacy_id/10;const int offset=legacy_id%10;
        m.bound=offset%2?IceRelationBound::lower:IceRelationBound::upper;
        if(offset<6){
            m.family=IceRelationFamily::row_gap;m.row=offset/2;m.index=m.row;
            m.reference_scale=IceRelationReferenceScale::nominal_pitch;
            if(m.column==0){
                // row_gap = pitch*(1+d[row]), pitch>0. The even legacy
                // relation is .01*pitch-pitch*d: it is the UPPER box face.
                m.canonical_kind=IceCanonicalKind::deviation_box;
                m.canonical_index=m.row;
                m.canonical_id=ice_deviation_box_canonical_id(m.row,m.bound);
                m.canonical_shape=ConstraintShape::affine;
                m.canonical_reference_scale=IceRelationReferenceScale::deviation_unit;
            }
        }else{
            m.family=IceRelationFamily::row_clearance;
            m.row=offset<8?0:3;m.side=offset<8?-1:1;
            m.reference_scale=IceRelationReferenceScale::nominal_row_clearance;
        }
    }else if(legacy_id<44){
        const int offset=legacy_id-20,house=offset/3;
        m.family=IceRelationFamily::ring_order;m.row=house%4;m.column=house/4;
        m.index=offset%3;m.strict=true;
        m.reference_scale=IceRelationReferenceScale::nominal_ring_outer_radius;
    }else if(legacy_id<186){
        // Lane blocks have 36 relations, except the final row has 34
        // because there is no following lane to compare against.
        m.row=(legacy_id-44)/36;int offset=legacy_id-(44+36*m.row);
        if(offset==0){
            m.family=IceRelationFamily::lane_house_order;m.strict=true;
            m.reference_scale=IceRelationReferenceScale::nominal_house_run;
        }else if(offset<=2){
            m.family=IceRelationFamily::lane_centre_slope;
            m.bound=offset==1?IceRelationBound::upper:IceRelationBound::lower;
            m.reference_scale=IceRelationReferenceScale::direction_unit;
        }else if(offset<=4){
            m.family=IceRelationFamily::lane_right_after_left;
            m.side=offset==3?1:-1;m.strict=true;
            m.reference_scale=IceRelationReferenceScale::nominal_house_run;
        }else if(offset<=6){
            m.family=IceRelationFamily::lane_transverse_after_left;
            m.side=offset==5?1:-1;
            m.reference_scale=IceRelationReferenceScale::nominal_house_run;
        }else if(offset<=8){
            m.family=IceRelationFamily::lane_right_after_transverse;
            m.side=offset==7?1:-1;m.strict=true;
            m.reference_scale=IceRelationReferenceScale::nominal_house_run;
        }else if(offset==9){
            m.family=IceRelationFamily::lane_centre_end_after_house;m.strict=true;
            m.reference_scale=IceRelationReferenceScale::nominal_house_run;
        }else if(offset==10){
            m.family=IceRelationFamily::lane_centre_end_before_right;m.strict=true;
            m.reference_scale=IceRelationReferenceScale::nominal_house_run;
        }else if(offset<=32){
            m.side=offset<22?-1:1;
            const int local=offset-(m.side<0?11:22);
            if(local==10){
                m.family=IceRelationFamily::lane_left_goal_support;
                m.reference_scale=IceRelationReferenceScale::nominal_house_run;
            }else{
                m.bound=local%2?IceRelationBound::lower:IceRelationBound::upper;
                switch(local/2){
                    case 0:m.family=IceRelationFamily::lane_right_x;break;
                    case 1:m.family=IceRelationFamily::lane_transverse_x;break;
                    case 2:m.family=IceRelationFamily::lane_side_slope;break;
                    case 3:m.family=IceRelationFamily::lane_width_left;break;
                    case 4:m.family=IceRelationFamily::lane_width_right;break;
                }
                m.reference_scale=local/2==2?IceRelationReferenceScale::direction_unit:
                    IceRelationReferenceScale::nominal_lane_width;
                if(m.family==IceRelationFamily::lane_width_left&&m.side>0){
                    // width(row,left) has no side dependence. Only these
                    // eight identities are merged; right endpoints differ.
                    m.canonical_id=legacy_id-11;m.canonical_index=m.canonical_id;
                }
            }
        }else if(offset==33){
            m.family=IceRelationFamily::lane_house_goal_support;
            m.reference_scale=IceRelationReferenceScale::nominal_house_run;
        }else{
            m.family=IceRelationFamily::lane_separation;
            m.index=offset-34; // 0: min(left[row],left[row+1]), 1: max(right.x)
            m.strict=true;m.reference_scale=IceRelationReferenceScale::nominal_pitch;
        }
    }else if(legacy_id<194){
        const int offset=legacy_id-186;
        m.family=IceRelationFamily::carrier_order;m.index=offset/2;
        m.side=offset%2?1:-1;m.strict=true;
        m.reference_scale=IceRelationReferenceScale::board_half_length;
    }else{
        const int offset=legacy_id-194;
        m.index=offset/3;m.column=m.index/2;
        m.side=m.index%2?1:-1;m.strict=true;
        switch(offset%3){
            case 0:
                m.family=IceRelationFamily::faceoff_goal_order;
                m.column=m.index<2?1:0;m.side=m.index==0||m.index==3?-1:1;
                m.reference_scale=IceRelationReferenceScale::nominal_goal_offset;break;
            case 1:
                m.family=IceRelationFamily::trapezoid_board_support;
                m.reference_scale=IceRelationReferenceScale::board_half_width;break;
            case 2:
                m.family=IceRelationFamily::trapezoid_goal_support;
                m.reference_scale=IceRelationReferenceScale::nominal_trapezoid_goal_span;break;
        }
    }
    if(m.canonical_reference_scale==IceRelationReferenceScale::invalid)
        m.canonical_reference_scale=m.reference_scale;
    return m;
}

RS_HD constexpr int ice_relation_canonical_id(int legacy_id){
    return ice_relation_meta(legacy_id).canonical_id;
}
RS_HD constexpr bool ice_relation_is_independent(int legacy_id){
    const auto m=ice_relation_meta(legacy_id);
    return m.valid()&&m.canonical_kind==IceCanonicalKind::relation&&m.canonical_id==legacy_id;
}
// Enumerate original IDs, including the requested relation itself. The caller
// can evaluate each untouched legacy margin; aliases never replace raw values.
template<class Visit> RS_HD inline int ice_relation_legacy_aliases(int legacy_id,Visit visit){
    const auto m=ice_relation_meta(legacy_id);if(!m.valid())return 0;
    int count=0;for(int id=0;id<ice_relation_legacy_count;++id)
        if(ice_relation_canonical_id(id)==m.canonical_id){visit(id);++count;}
    return count;
}
} // namespace rinkspace
