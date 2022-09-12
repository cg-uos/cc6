#version 330 core

in vec2 vTexCoord;

uniform sampler2D   tex_back;
uniform sampler2D   tex_front;
uniform sampler3D   tex_volume;
uniform sampler2D   tex_colormap;
uniform sampler2D   tex_colormap_2d;
uniform vec3        scale_norm;
uniform vec3        offset_norm;
uniform vec3        scale_lattice;
uniform vec3        offset_lattice;
uniform vec3        dim;
uniform float       dim_max;
uniform float       level;
uniform mat4        MV;
uniform float       scale_step;
uniform float       scale_delta;

out vec4 fColor;

struct return_t
{
    vec4        gradient;
    float[6]    d2;
};

#define MAX_ITERATIONS 1000


#define DENOM_M 0.00260416667 // 1/384
#define DENOM_G 0.015625      // 1/64
#define DENOM_H 0.0625        // 1/16

#define TYPE_BLUE   0
#define TYPE_GREEN  1
#define TYPE_RED    2

vec4  u;
float u2000, u3000, u0200, u0300, u0020, u0030, u0002, u0003;

#define u0  u[0]
#define u1  u[1]
#define u2  u[2]
#define u3  u[3]

ivec3 org;
ivec3 type_R;
int	  type_P;
int	  type_tet;
int	  idx;

float[38]	c;

#define c0	c[0]		//(-1,-1, 0)
#define c1	c[1]		//(-1, 0,-1)
#define c2	c[2]		//(-1, 0, 0)
#define c3	c[3]		//(-1, 0, 1)
#define c4	c[4]		//(-1, 1, 0)
#define c5	c[5]		//(-1, 1, 1)
#define c6	c[6]		//( 0,-1,-1)
#define c7	c[7]		//( 0,-1, 0)
#define c8	c[8]		//( 0,-1, 1)
#define c9	c[9]		//( 0, 0,-1)
#define c10	c[10]		//( 0, 0, 0)
#define c11	c[11]		//( 0, 0, 1)
#define c12	c[12]		//( 0, 0, 2)
#define c13	c[13]		//( 0, 1,-1)
#define c14	c[14]		//( 0, 1, 0)
#define c15	c[15]		//( 0, 1, 1)
#define c16	c[16]		//( 0, 1, 2)
#define c17	c[17]		//( 0, 2, 0)
#define c18	c[18]		//( 0, 2, 1)
#define c19	c[19]		//( 1,-1,-1)
#define c20	c[20]		//( 1,-1, 0)
#define c21	c[21]		//( 1,-1, 1)
#define c22	c[22]		//( 1, 0,-1)
#define c23	c[23]		//( 1, 0, 0)
#define c24	c[24]		//( 1, 0, 1)
#define c25	c[25]		//( 1, 0, 2)
#define c26	c[26]		//( 1, 1,-1)
#define c27	c[27]		//( 1, 1, 0)
#define c28	c[28]		//( 1, 1, 1)
#define c29	c[29]		//( 1, 1, 2)
#define c30	c[30]		//( 1, 2, 0)
#define c31	c[31]		//( 1, 2, 1)
#define c32	c[32]		//( 2,-1, 0)
#define c33	c[33]		//( 2, 0,-1)
#define c34	c[34]		//( 2, 0, 0)
#define c35	c[35]		//( 2, 0, 1)
#define c36	c[36]		//( 2, 1, 0)
#define c37	c[37]		//( 2, 1, 1)


#define EVAL(p) (preprocess(p), fetch_coefficients(), eval_M())

#define	GET_DATA(texcoords)	texelFetch(tex_volume, texcoords, 0).r

void preprocess(vec3 p_in)
{
	// Find the nearest lattice point of p_in
	org = ivec3(round(p_in));

	// Compute local coordinates
	vec3	p_local = p_in - vec3(org);

	// Find the sign change matrix R
	type_R = 2*ivec3(p_local.x>0, p_local.y>0, p_local.z>0)-1;

	// Transform p_local into the (+,+,+) octant
	vec3	p_cube = p_local.xyz*vec3(type_R);


	// Find the even permutation matrix P
	ivec4	bit = ivec4( p_cube[0]-p_cube[1]-p_cube[2]>0,
						-p_cube[0]+p_cube[1]-p_cube[2]>0,
						-p_cube[0]-p_cube[1]+p_cube[2]>0,
						 p_cube[0]+p_cube[1]+p_cube[2]>1);
	// bit_tet   type_tet type_P permutation
	// 0 1 2 3
	// -------------------------------------
	// 1 0 0 0       2      0        123    (edge/red)  
	// 0 1 0 0       2      1        231    (edge/red)
	// 0 0 1 0       2      2        312    (edge/red)
	// 0 0 0 1       0      0        123    (oct/blue)
	// 0 0 0 0       1      0        123    (vert/green)
	type_tet = (1+bit[3])*(bit[0]+bit[1]+bit[2]) + (1-bit[3]);	// 0 (oct), 1 (vert), 2 (edge)
	type_P = bit[1] + 2*bit[2];	// one of three even permutations

	// Transform p_cube into the reference tetrahedron(Fig. 5(c))
	vec4	p_ref = vec4(p_cube[type_P],
						 p_cube[(type_P+1)%3],
						 p_cube[(type_P+2)%3], 1);
    
    // Compute the barycentric coordinates.
    u = float(type_tet==TYPE_BLUE)*2.0
        *vec4( p_ref.x+p_ref.y+p_ref.z-1  ,
                      -p_ref.y        +0.5,
                              -p_ref.z+0.5,
              -p_ref.x                +0.5)
        +float(type_tet==TYPE_GREEN)
            *vec4(-p_ref.x-p_ref.y-p_ref.z    +1,
                   p_ref.x-p_ref.y+p_ref.z      ,
                   p_ref.x+p_ref.y-p_ref.z      ,             
                  -p_ref.x+p_ref.y+p_ref.z      )
        +float(type_tet==TYPE_RED)*2.0
            *vec4(-p_ref.x                +0.5,
                                   p_ref.z    ,
                                   p_ref.y    ,
                   p_ref.x-p_ref.y-p_ref.z    );

    u2000 = u0*u0;
    u3000 = u0*u2000;
    u0200 = u1*u1;
    u0300 = u1*u0200;
    u0020 = u2*u2;
    u0030 = u2*u0020;
    u0002 = u3*u3;
    u0003 = u3*u0002;
}

void fetch_coefficients(void)
{
	ivec3 bit_P = ivec3(type_P==0, type_P==1, type_P==2);
	ivec3 dirx = ivec3(type_R.x*bit_P.x, type_R.y*bit_P.y, type_R.z*bit_P.z);
	ivec3 diry = ivec3(type_R.x*bit_P.z, type_R.y*bit_P.x, type_R.z*bit_P.y);
	ivec3 dirz = ivec3(type_R.x*bit_P.y, type_R.y*bit_P.z, type_R.z*bit_P.x);

	ivec3	coords = org;
#define	FETCH_C(idx_c, offset)	coords += (offset); c[idx_c] = GET_DATA(coords);
	c[10] = GET_DATA(coords);
	FETCH_C(23, dirx);
	FETCH_C(22, -dirz);
	FETCH_C(33, dirx);
	FETCH_C(34, dirz);
	FETCH_C(32, -diry);
	FETCH_C(20, -dirx);
	FETCH_C(19, -dirz);
	FETCH_C(6, -dirx);
	FETCH_C(7, dirz);
	FETCH_C(0, -dirx);
	FETCH_C(2, diry);
	FETCH_C(1, -dirz);
	FETCH_C(9, dirx);
	FETCH_C(13, diry);
	FETCH_C(26, dirx);
	FETCH_C(27, dirz);
	FETCH_C(36, dirx);
	FETCH_C(4, -3*dirx);
	FETCH_C(14, dirx);
	FETCH_C(17, diry);
	FETCH_C(30, dirx);
	FETCH_C(31, dirz);
	FETCH_C(18, -dirx);
	FETCH_C(15, -diry);
	FETCH_C(5, -dirx);
	FETCH_C(3, -diry);
	FETCH_C(11, dirx);
	FETCH_C(8, -diry);
	FETCH_C(21, dirx);
	FETCH_C(24, diry);
	FETCH_C(35, dirx);
	FETCH_C(37, diry);
	FETCH_C(28, -dirx);
	FETCH_C(29, dirz);
	FETCH_C(16, -dirx);
	FETCH_C(12, -diry);
	FETCH_C(25, dirx);
#undef	FETCH_C
}

float eval_M_expr_red(void)
{
    return 8*u3000*((c0+c1+c3+c4+c6+c8+c13+c15+c20+c22+c24+c27) + 4*(c2+c7+c9+c11+c14+c23) + 12*c10) +
           4*(u0300*((c2+c3+c9+c12+c22+c25+c34+c35) + 4*(c7+c8+c14+c15+c20+c21+c27+c28) + 14*(c10+c11+c23+c24)) +
              u0030*((c2+c4+c7+c17+c20+c30+c34+c36) + 4*(c9+c11+c13+c15+c22+c24+c26+c28) + 14*(c10+c14+c23+c27))) +
             u0003*((c0+c1+c3+c4+c32+c33+c35+c36) + 4*(c2+c6+c8+c13+c15+c19+c21+c26+c28+c34) + 23*(c7+c9+c11+c14+c20+c22+c24+c27) + 76*(c10+c23)) +
           3*u0002*(2*u0*((c0+c1+c3+c4+c19+c21+c26+c28) + 3*(c6+c8+c13+c15) + 4*c2 + 9*(c20+c22+c24+c27) + 14*(c7+c9+c11+c14) + 32*c23 + 44*c10) +
                      u1*((c0+c4+c6+c13+c19+c26+c32+c36) + 2*(c3+c35) + 4*(c2+c34) + 7*(c8+c15+c21+c28) + 14*(c9+c22) + 23*(c7+c14+c20+c27) + 32*(c11+c24) + 76*(c10+c23)) +
                      u2*((c1+c3+c6+c8+c19+c21+c33+c35) + 2*(c4+c36) + 4*(c2+c34) + 7*(c13+c15+c26+c28) + 14*(c7+c20) + 23*(c9+c11+c22+c24) + 32*(c14+c27) + 76*(c10+c23))) +
          12*(u2000*(u1*((c0+c4+c6+c13) + 2*(c3+c22) + 3*(c8+c15+c20+c27) + 4*(c2+c9+c24) + 8*(c7+c14) + 12*(c11+c23) + 24*c10) +
                     u2*((c1+c3+c6+c8) + 2*(c4+c20) + 3*(c13+c15+c22+c24) + 4*(c2+c7+c27) + 8*(c9+c11) + 12*(c14+c23) + 24*c10) +
                     u3*((c0+c1+c3+c4) + 2*(c6+c8+c13+c15) + 3*(c20+c22+c24+c27) + 4*c2 + 8*(c7+c9+c11+c14) + 12*c23 + 24*c10)) +
              u0200*(2*u0*((c2+c3+c9+c21+c22+c28) + 2*(c8+c15+c20+c27) + 3*(c7+c14) + 4*c24 + 7*(c11+c23) + 10*c10) +
                       u2*((c2+c3+c8+c21+c34+c35) + 2*(c9+c22) + 3*(c7+c20) + 5*(c15+c28) + 7*(c14+c27) + 11*(c11+c24) + 17*(c10+c23)) +
                       u3*((c2+c3+c34+c35) + 2*(c9+c22) + 3*(c8+c15+c21+c28) + 5*(c7+c14+c20+c27) + 11*(c11+c24) + 17*(c10+c23))) +
              u0020*(2*u0*((c2+c4+c7+c20+c26+c28) + 2*(c13+c15+c22+c24) + 3*(c9+c11) + 4*c27 + 7*(c14+c23) + 10*c10) +
                       u1*((c2+c4+c13+c26+c34+c36) + 2*(c7+c20) + 3*(c9+c22) + 5*(c15+c28) + 7*(c11+c24) + 11*(c14+c27) + 17*(c10+c23)) +
                       u3*((c2+c4+c34+c36) + 2*(c7+c20) + 3*(c13+c15+c26+c28) + 5*(c9+c11+c22+c24) + 11*(c14+c27) + 17*(c10+c23))) +
              u3*(u0*u1*((c0+c4+c6+c13) + 2*(c3+c21+c28) + 4*c2 + 5*(c8+c15) + 6*c22 + 8*c9 + 9*(c20+c27) + 12*c24 + 14*(c7+c14) + 20*c11 + 32*c23 + 44*c10)+
                  u0*u2*((c1+c3+c6+c8) + 2*(c4+c26+c28) + 4*c2 + 5*(c13+c15) + 6*c20 + 8*c7 + 9*(c22+c24) + 12*c27 + 14*(c9+c11) + 20*c14 + 32*c23 + 44*c10) +
                  u1*u2*((c3+c4+c8+c13+c21+c26+c35+c36) + 2*(c2+c34) + 6*(c15+c28) + 7*(c7+c9+c20+c22) + 16*(c11+c14+c24+c27) + 38*(c10+c23)))) +
          24*u0*u1*u2*((c3+c4+c8+c13) + 2*(c2+c28) + 3*(c20+c22) + 4*(c7+c9+c15) + 6*(c24+c27) + 10*(c11+c14) + 16*c23 + 22*c10);
}

float eval_M_expr_green(void)
{
    return   8*u3000*((c0+c1+c3+c4+c6+c8+c13+c15+c20+c22+c24+c27) + 4*(c2+c7+c9+c11+c14+c23) + 12*c10) +
             4*(u0300*((c2+c3+c9+c12+c22+c25+c34+c35) + 4*(c7+c8+c14+c15+c20+c21+c27+c28) + 14*(c10+c11+c23+c24)) +
                u0030*((c2+c4+c7+c17+c20+c30+c34+c36) + 4*(c9+c11+c13+c15+c22+c24+c26+c28) + 14*(c10+c14+c23+c27)) +
                u0003*((c7+c8+c9+c12+c13+c16+c17+c18) + 4*(c2+c3+c4+c5+c23+c24+c27+c28) + 14*(c10+c11+c14+c15))) +
            12*(u2000*(u1*((c0+c4+c6+c13) + 2*(c3+c22) + 3*(c8+c15+c20+c27) + 4*(c2+c9+c24) + 8*(c7+c14) + 12*(c11+c23) + 24*c10) +
                       u2*((c1+c3+c6+c8) + 2*(c4+c20) + 3*(c13+c15+c22+c24) + 4*(c2+c7+c27) + 8*(c9+c11) + 12*(c14+c23) + 24*c10) +
                       u3*((c0+c1+c20+c22) + 2*(c8+c13) + 3*(c3+c4+c24+c27) + 4*(c7+c9+c15) + 8*(c2+c23) + 12*(c11+c14) + 24*c10)) +
                u0200*(2*u0*((c2+c3+c9+c21+c22+c28) + 2*(c8+c15+c20+c27) + 3*(c7+c14) + 4*c24 + 7*(c11+c23) + 10*c10) +
                         u2*((c2+c3+c8+c21+c34+c35) + 2*(c9+c22) + 3*(c7+c20) + 5*(c15+c28) + 7*(c14+c27) + 11*(c11+c24) + 17*(c10+c23)) +
                         u3*((c9+c12+c20+c21+c22+c25) + 2*(c2+c3) + 3*(c7+c8) + 5*(c27+c28) + 7*(c14+c15) + 11*(c23+c24) + 17*(c10+c11))) +
                u0020*(2*u0*((c2+c4+c7+c20+c26+c28) + 2*(c13+c15+c22+c24) + 3*(c9+c11) + 4*c27 + 7*(c14+c23) + 10*c10) +
                         u1*((c2+c4+c13+c26+c34+c36) + 2*(c7+c20) + 3*(c9+c22) + 5*(c15+c28) + 7*(c11+c24) + 11*(c14+c27) + 17*(c10+c23)) +
                         u3*((c7+c17+c20+c22+c26+c30) + 2*(c2+c4) + 3*(c9+c13) + 5*(c24+c28) + 7*(c11+c15) + 11*(c23+c27) + 17*(c10+c14))) +
                u0002*(2*u0*((c5+c7+c8+c9+c13+c28) + 2*(c3+c4+c24+c27) + 3*(c2+c23) + 4*c15 + 7*(c11+c14) + 10*c10) +
                         u1*((c4+c5+c9+c12+c13+c16) + 2*(c7+c8) + 3*(c2+c3) + 5*(c27+c28) + 7*(c23+c24) + 11*(c14+c15) + 17*(c10+c11)) +
                         u2*((c3+c5+c7+c8+c17+c18) + 2*(c9+c13) + 3*(c2+c4) + 5*(c24+c28) + 7*(c23+c27) + 11*(c11+c15) + 17*(c10+c14)))) +
            24*(u0*(u1*u3*((c4+c13+c20+c22) + 2*(c9+c28) + 3*(c3+c8) + 4*(c2+c7+c27) + 6*(c15+c24) + 10*(c14+c23) + 16*c11 + 22*c10) +
                    u1*u2*((c3+c4+c8+c13) + 2*(c2+c28) + 3*(c20+c22) + 4*(c7+c9+c15) + 6*(c24+c27) + 10*(c11+c14) + 16*c23 + 22*c10) +
                    u2*u3*((c3+c8+c20+c22) + 2*(c7+c28) + 3*(c4+c13) + 4*(c2+c9+c24) + 6*(c15+c27) + 10*(c11+c23) + 16*c14 + 22*c10)) +
                u1*u2*u3*((c3+c4+c8+c13+c20+c22) + 2*(c2+c7+c9) + 6*c28 + 8*(c15+c24+c27) + 12*(c11+c14+c23) + 18*c10));
}

float eval_M_expr_blue(void)
{
    return   2*u3000*((c2+c3+c4+c5+c7+c8+c9+c12+c13+c16+c17+c18+c20+c21+c22+c25+c26+c29+c30+c31+c34+c35+c36+c37) + 21*(c10+c11+c14+c15+c23+c24+c27+c28)) +
             4*(u0300*((c2+c3+c9+c12+c22+c25+c34+c35) + 4*(c7+c8+c14+c15+c20+c21+c27+c28) + 14*(c10+c11+c23+c24)) +
                u0030*((c2+c4+c7+c17+c20+c30+c34+c36) + 4*(c9+c11+c13+c15+c22+c24+c26+c28) + 14*(c10+c14+c23+c27)) +
                u0003*((c7+c8+c9+c12+c13+c16+c17+c18) + 4*(c2+c3+c4+c5+c23+c24+c27+c28) + 14*(c10+c11+c14+c15))) +
             3*u2000*(u1*((c4+c5+c13+c16+c26+c29+c36+c37) + 3*(c2+c3+c9+c12+c22+c25+c34+c35) + 4*(c7+c8+c20+c21) + 34*(c14+c15+c27+c28) + 50*(c10+c11+c23+c24)) +
                      u2*((c3+c5+c8+c18+c21+c31+c35+c37) + 3*(c2+c4+c7+c17+c20+c30+c34+c36) + 4*(c9+c13+c22+c26) + 34*(c11+c15+c24+c28) + 50*(c10+c14+c23+c27)) +
                      u3*((c20+c21+c22+c25+c26+c29+c30+c31) + 3*(c7+c8+c9+c12+c13+c16+c17+c18) + 4*(c2+c3+c4+c5) + 34*(c23+c24+c27+c28) + 50*(c10+c11+c14+c15))) +
            12*(u0200*(u0*((c2+c3+c9+c12+c22+c25+c34+c35) + 2*(c7+c8+c20+c21) + 6*(c14+c15+c27+c28) + 14*(c10+c11+c23+c24)) +
                       u2*((c2+c3+c8+c21+c34+c35) + 2*(c9+c22) + 3*(c7+c20) + 5*(c15+c28) + 7*(c14+c27) + 11*(c11+c24) + 17*(c10+c23)) +
                       u3*((c9+c12+c20+c21+c22+c25) + 2*(c2+c3) + 3*(c7+c8) + 5*(c27+c28) + 7*(c14+c15) + 11*(c23+c24) + 17*(c10+c11))) +
                u0020*(u0*((c2+c4+c7+c17+c20+c30+c34+c36) + 2*(c9+c13+c22+c26) + 6*(c11+c15+c24+c28) + 14*(c10+c14+c23+c27)) +
                       u1*((c2+c4+c13+c26+c34+c36) + 2*(c7+c20) + 3*(c9+c22) + 5*(c15+c28) + 7*(c11+c24) + 11*(c14+c27) + 17*(c10+c23)) +
                       u3*((c7+c17+c20+c22+c26+c30) + 2*(c2+c4) + 3*(c9+c13) + 5*(c24+c28) + 7*(c11+c15) + 11*(c23+c27) + 17*(c10+c14))) +
                u0002*(u0*((c7+c8+c9+c12+c13+c16+c17+c18) + 2*(c2+c3+c4+c5) + 6*(c23+c24+c27+c28) + 14*(c10+c11+c14+c15)) +
                       u1*((c4+c5+c9+c12+c13+c16) + 2*(c7+c8) + 3*(c2+c3) + 5*(c27+c28) + 7*(c23+c24) + 11*(c14+c15) + 17*(c10+c11)) +
                       u2*((c3+c5+c7+c8+c17+c18) + 2*(c9+c13) + 3*(c2+c4) + 5*(c24+c28) + 7*(c23+c27) + 11*(c11+c15) + 17*(c10+c14))) + 
                u0*(u1*u2*((c3+c4+c8+c13+c21+c26+c35+c36) + 2*(c2+c34) + 3*(c7+c9+c20+c22) + 14*(c15+c28) + 20*(c11+c14+c24+c27) + 30*(c10+c23)) +
                    u1*u3*((c4+c5+c13+c16+c20+c21+c22+c25) + 2*(c9+c12) + 3*(c2+c3+c7+c8) + 14*(c27+c28) + 20*(c14+c15+c23+c24) + 30*(c10+c11)) +
                    u2*u3*((c3+c5+c8+c18+c20+c22+c26+c30) + 2*(c7+c17) + 3*(c2+c4+c9+c13) + 14*(c24+c28) + 20*(c11+c15+c23+c27) + 30*(c10+c14)))) +
            24*u1*u2*u3*((c3+c4+c8+c13+c20+c22) + 2*(c2+c7+c9) + 6*c28 + 8*(c15+c24+c27) + 12*(c11+c14+c23) + 18*c10);

}

float eval_M(void)
{
    return (float(type_tet==TYPE_BLUE)*eval_M_expr_blue()
           +float(type_tet==TYPE_GREEN)*eval_M_expr_green()
           +float(type_tet==TYPE_RED)*eval_M_expr_red()) * DENOM_M;
}
vec3 eval_G_expr_red(void)
{
    vec3 g;

    g.x = u0002*((-c1-c3-c4-c0+c32+c33+c35+c36) + 2*(c19-c6-c8-c13-c15+c21+c26+c28) + 4*(-c2+c34) + 5*(-c7-c9-c11-c14+c20+c22+c24+c27) + 12*(-c10+c23)) +
          4*(u2000*((c20-c1-c3-c4-c0+c22+c24+c27) + 4*(-c2+c23)) +
             u0200*((c20-c3-c7-c8-c14-c15-c2+c21+c27+c28+c34+c35) + 3*(-c10-c11+c23+c24)) +
             u0020*((c22-c4-c9-c11-c13-c15-c2+c24+c26+c28+c34+c36) + 3*(-c10-c14+c23+c27)) +
             u0*(u1*((-c4-c6-c8-c13-c15-c0) + 2*(-c3-c7-c14+c21+c22+c28) + 3*(c20+c27) + 4*(-c2-c10-c11+c24) + 8*c23) +
                 u2*((-c3-c6-c8-c13-c15-c1) + 2*(c20-c4-c9-c11+c26+c28) + 3*(c22+c24) + 4*(-c2-c10-c14+c27) + 8*c23) +
                 u3*((c19-c1-c3-c4-c6-c8-c13-c15-c0+c21+c26+c28) + 2*(-c7-c9-c11-c14) + 3*(c20+c22+c24+c27) + 4*(-c2-c10) + 8*c23)) +
             u1*u2*((c20-c3-c4-c7-c8-c9-c13+c21+c22+c26+c35+c36) + 2*(-c15-c2+c28+c34) + 4*(-c11-c14+c24+c27) + 6*(-c10+c23))) +
          2*u3*(u1*((c19-c4-c6-c13-c0+c26+c32+c36) + 2*(-c3-c9+c22+c35) + 3*(-c8-c15+c21+c28) + 4*(-c2+c34) + 5*(-c7-c14+c20+c27) + 8*(-c11+c24) + 12*(-c10+c23)) +
                u2*((c19-c3-c6-c8-c1+c21+c33+c35) + 2*(-c4-c7+c20+c36) + 3*(-c13-c15+c26+c28) + 4*(-c2+c34) + 5*(-c9-c11+c22+c24) + 8*(-c14+c27) + 12*(-c10+c23)));

    g.y = u0002*((c4-c0-c32+c36) + 3*(-c6-c8+c13+c15-c19-c21+c26+c28) + 9*(-c7+c14-c20+c27)) +
          4*(  u2000*((c4-c0-c6-c8+c13+c15-c20+c27) + 4*(-c7+c14)) +
             2*u0200*(c14-c8-c7+c15-c20-c21+c27+c28) +
               u0020*((c13-c9-c11-c7+c15+c17-c20-c22-c24+c26+c28+c30) + 3*(-c10+c14-c23+c27)) +
               u0*(u1*((c4-c0-c6+c13) + 2*(-c21+c28) + 3*(-c8+c15-c20+c27) + 6*(-c7+c14)) +
                   u2*((-c3-c1-c6-c8-c22-c24) + 2*(c4-c9-c11-c20+c26+c28) + 3*(c13+c15) + 4*(-c7-c10-c23+c27) + 8*c14) +
                   u3*((c4-c0-c19-c21+c26+c28) + 2*(-c6-c8+c13+c15) + 3*(-c20+c27) + 6*(-c7+c14))) +
               u1*u2*((c4-c3-c8-c9+c13-c21-c22+c26-c35+c36) + 2*(-c11-c24) + 3*(-c7-c20) + 4*(-c10+c15-c23+c28) + 6*(c14+c27))) +
          2*u3*(u1*((c4-c0-c6+c13-c19+c26-c32+c36) + 5*(-c8+c15-c21+c28) + 9*(-c7+c14-c20+c27)) +
                u2*((-c3-c1-c6-c8-c19-c21-c33-c35) + 2*(c4+c36) + 3*(-c9-c11-c22-c24) + 5*(c13+c15+c26+c28) + 6*(-c7-c20) + 8*(-c10-c23) + 12*(c14+c27)));

    g.z = u0002*((c3-c1-c33+c35) + 3*(-c6+c8-c13+c15-c19+c21-c26+c28) + 9*(-c9+c11-c22+c24)) +
          4*(  u2000*((c3-c1-c6+c8-c13+c15-c22+c24) + 4*(-c9+c11)) +
               u0200*((c8-c7-c9+c12-c14+c15-c20+c21-c22+c25-c27+c28) + 3*(-c10+c11-c23+c24)) +
             2*u0020*((c11-c9-c13+c15-c22+c24-c26+c28)) +
               u0*(u1*((-c0-c4-c6-c13-c20-c27) + 2*(c3-c7-c14+c21-c22+c28) + 3*(c8+c15) + 4*(-c9-c10-c23+c24) + 8*c11) +
                   u2*((c3-c1-c6+c8) + 2*(-c26+c28) + 3*(-c13+c15-c22+c24) + 6*(-c9+c11)) +
                   u3*((c3-c1-c19+c21-c26+c28) + 2*(-c6+c8-c13+c15) + 3*(-c22+c24) + 6*(-c9+c11))) +
               u1*u2*((c3-c4-c7+c8-c13-c20+c21-c26+c35-c36) + 2*(-c14-c27) + 3*(-c9-c22) + 4*(-c10+c15-c23+c28) + 6*(c11+c24))) +
          2*u3*(u1*((-c0-c4-c6-c13-c19-c26-c32-c36) + 2*(c3+c35) + 3*(-c7-c14-c20-c27) + 5*(c8+c15+c21+c28) + 6*(-c9-c22) + 8*(-c10-c23) + 12*(c11+c24)) +
                u2*((c3-c1-c6+c8-c19+c21-c33+c35) + 5*(-c13+c15-c26+c28) + 9*(-c9+c11-c22+c24)));

    return g;
}


vec3 eval_G_expr_green(void)
{
    vec3 g;

    g.x = 4*(  u2000*((c20-c1-c3-c4-c0+c22+c24+c27) + 4*(-c2+c23)) +
               u0200*((c20-c3-c7-c8-c14-c15-c2+c21+c27+c28+c34+c35) + 3*(-c10-c11+c23+c24)) +
               u0020*((c22-c4-c9-c11-c13-c15-c2+c24+c26+c28+c34+c36) + 3*(-c10-c14+c23+c27)) +
             2*u0002*((c23-c3-c4-c5-c2+c24+c27+c28)) +
               u0*(u1*((-c4-c6-c8-c13-c15-c0) + 2*(-c3-c7-c14+c21+c22+c28) + 3*(c20+c27) + 4*(-c2-c10-c11+c24) + 8*c23) +
                   u2*((-c3-c6-c8-c13-c15-c1) + 2*(c20-c4-c9-c11+c26+c28) + 3*(c22+c24) + 4*(-c2-c10-c14+c27) + 8*c23) +
                   u3*((c20-c1-c0+c22) + 2*(-c5+c28) + 3*(-c3-c4+c24+c27) + 6*(-c2+c23))) +
               u1*(u2*((c20-c3-c4-c7-c8-c9-c13+c21+c22+c26+c35+c36) + 2*(-c15-c2+c28+c34) + 4*(-c11-c14+c24+c27) + 6*(-c10+c23)) +
                   u3*((c20-c4-c5-c7-c8-c13-c16+c21+c22+c25) + 2*(-c14-c15) + 3*(-c3-c2) + 4*(-c10-c11+c27+c28) + 6*(c23+c24))) +
               u2*u3*((c20-c3-c5-c8-c9-c13-c18+c22+c26+c30) + 2*(-c11-c15) + 3*(-c4-c2) + 4*(-c10-c14+c24+c28) + 6*(c23+c27)));


    g.y = 4*(  u2000*((c4-c0-c6-c8+c13+c15-c20+c27) + 4*(-c7+c14)) +
             2*u0200*((c14-c8-c7+c15-c20-c21+c27+c28)) +
               u0020*((c13-c9-c11-c7+c15+c17-c20-c22-c24+c26+c28+c30) + 3*(-c10+c14-c23+c27)) +
               u0002*((c4-c3-c2+c5-c7-c8+c17+c18-c23-c24+c27+c28) + 3*(-c10-c11+c14+c15)) +
               u0*(u1*((c4-c0-c6+c13) + 2*(-c21+c28) + 3*(-c8+c15-c20+c27) + 6*(-c7+c14)) +
                   u2*((-c3-c1-c6-c8-c22-c24) + 2*(c4-c9-c11-c20+c26+c28) + 3*(c13+c15) + 4*(-c7-c10-c23+c27) + 8*c14) +
                   u3*((-c1-c3-c0-c20-c22-c24) + 2*(-c2+c5-c8+c13-c23+c28) + 3*(c4+c27) + 4*(-c7-c10-c11+c15) + 8*c14)) +
               u1*(u2*((c4-c3-c8-c9+c13-c21-c22+c26-c35+c36) + 2*(-c11-c24) + 3*(-c7-c20) + 4*(-c10+c15-c23+c28) + 6*(c14+c27)) +
                   u3*((c4-c3-c2+c5+c13+c16-c20-c21-c22-c25) + 2*(-c23-c24) + 3*(-c7-c8) + 4*(-c10-c11+c27+c28) + 6*(c14+c15))) +
               u2*u3*((c4-c3-c2+c5-c8-c9+c13+c18-c20-c22+c26+c30) + 2*(-c7+c17-c24+c28) + 4*(-c11+c15-c23+c27) + 6*(-c10+c14)));

    g.z = 4*(  u2000*((c3-c1-c6+c8-c13+c15-c22+c24) + 4*(-c9+c11)) +
               u0200*((c8-c7-c9+c12-c14+c15-c20+c21-c22+c25-c27+c28) + 3*(-c10+c11-c23+c24)) +
               u0002*((c3-c2-c4+c5-c9+c12-c13+c16-c23+c24-c27+c28) + 3*(-c10+c11-c14+c15)) +
             2*u0020*((c11-c9-c13+c15-c22+c24-c26+c28)) +
             u0*(u1*((-c0-c4-c6-c13-c20-c27) + 2*(c3-c7-c14+c21-c22+c28) + 3*(c8+c15) + 4*(-c9-c10-c23+c24) + 8*c11) +
                 u2*((c3-c1-c6+c8) + 2*(-c26+c28) + 3*(-c13+c15-c22+c24) + 6*(-c9+c11)) +
                 u3*((-c1-c0-c4-c20-c22-c27) + 2*(-c2+c5+c8-c13-c23+c28) + 3*(c3+c24) + 4*(-c9-c10-c14+c15) + 8*c11)) +
             u1*(u2*((c3-c4-c7+c8-c13-c20+c21-c26+c35-c36) + 2*(-c14-c27) + 3*(-c9-c22) + 4*(-c10+c15-c23+c28) + 6*(c11+c24)) +
                 u3*((c3-c2-c4+c5-c7+c8-c13+c16-c20+c21-c22+c25) + 2*(-c9+c12-c27+c28) + 4*(-c14+c15-c23+c24) + 6*(-c10+c11))) +
             u2*u3*((c3-c2-c4+c5+c8+c18-c20-c22-c26-c30) + 2*(-c23-c27) + 3*(-c9-c13) + 4*(-c10-c14+c24+c28) + 6*(c11+c15)));

    return g;
}


vec3 eval_G_expr_blue(void)
{
    vec3 g;

    g.x =   u2000*((c20-c7-c8-c9-c12-c13-c16-c17-c18+c21+c22+c25+c26+c29+c30+c31) + 2*(-c3-c4-c5-c2+c34+c35+c36+c37) + 8*(-c10-c11-c14-c15+c23+c24+c27+c28)) +
          8*u0002*((c23-c3-c4-c5-c2+c24+c27+c28)) +
          4*(u0200*((c20-c3-c7-c8-c14-c15-c2+c21+c27+c28+c34+c35) + 3*(-c10-c11+c23+c24)) +
             u0020*((c22-c4-c9-c11-c13-c15-c2+c24+c26+c28+c34+c36) + 3*(-c10-c14+c23+c27)) +
             u1*(u2*((c20-c3-c4-c7-c8-c9-c13+c21+c22+c26+c35+c36) + 2*(-c15-c2+c28+c34) + 4*(-c11-c14+c24+c27) + 6*(-c10+c23)) +
                 u3*((c20-c4-c5-c7-c8-c13-c16+c21+c22+c25) + 2*(-c14-c15) + 3*(-c3-c2) + 4*(-c10-c11+c27+c28) + 6*(c23+c24))) +
             u2*u3*((c20-c3-c5-c8-c9-c13-c18+c22+c26+c30) + 2*(-c11-c15) + 3*(-c4-c2) + 4*(-c10-c14+c24+c28) + 6*(c23+c27))) +
          2*u0*(u1*((-c4-c5-c9-c12-c13-c16+c22+c25+c26+c29+c36+c37) + 2*(c20-c7-c8+c21) + 3*(-c3-c2+c34+c35) + 6*(-c14-c15+c27+c28) + 10*(-c10-c11+c23+c24)) +
                u2*((c20-c3-c5-c7-c8-c17-c18+c21+c30+c31+c35+c37) + 2*(-c9-c13+c22+c26) + 3*(-c4-c2+c34+c36) + 6*(-c11-c15+c24+c28) + 10*(-c10-c14+c23+c27)) +
                u3*((c20-c7-c8-c9-c12-c13-c16-c17-c18+c21+c22+c25+c26+c29+c30+c31) + 4*(-c3-c4-c5-c2) + 6*(-c10-c11-c14-c15) + 10*(c23+c24+c27+c28)));

    g.y =   u2000*((c4-c3-c2+c5-c9-c12+c13+c16-c22-c25+c26+c29-c34-c35+c36+c37) + 2*(-c7-c8+c17+c18-c20-c21+c30+c31) + 8*(-c10-c11+c14+c15-c23-c24+c27+c28)) +
          8*u0200*((c14-c8-c7+c15-c20-c21+c27+c28)) +
          4*(u0020*((c13-c9-c11-c7+c15+c17-c20-c22-c24+c26+c28+c30) + 3*(-c10+c14-c23+c27)) +
             u0002*((c4-c3-c2+c5-c7-c8+c17+c18-c23-c24+c27+c28) + 3*(-c10-c11+c14+c15)) +
             u1*(u2*((c4-c3-c8-c9+c13-c21-c22+c26-c35+c36) + 2*(-c11-c24) + 3*(-c7-c20) + 4*(-c10+c15-c23+c28) + 6*(c14+c27)) +
                 u3*((c4-c3-c2+c5+c13+c16-c20-c21-c22-c25) + 2*(-c23-c24) + 3*(-c7-c8) + 4*(-c10-c11+c27+c28) + 6*(c14+c15))) +
             u2*u3*((c4-c3-c2+c5-c8-c9+c13+c18-c20-c22+c26+c30) + 2*(-c7+c17-c24+c28) + 4*(-c11+c15-c23+c27) + 6*(-c10+c14))) +
          2*u0*(u1*((c4-c3-c2+c5-c9-c12+c13+c16-c22-c25+c26+c29-c34-c35+c36+c37) + 4*(-c7-c8-c20-c21) + 6*(-c10-c11-c23-c24) + 10*(c14+c15+c27+c28)) +
                u2*((c4-c3-c2+c5-c8+c18-c21+c31-c34-c35+c36+c37) + 2*(-c9+c13-c22+c26) + 3*(-c7+c17-c20+c30) + 6*(-c11+c15-c24+c28) + 10*(-c10+c14-c23+c27)) +
                u3*((-c9-c12+c13+c16-c20-c21-c22-c25+c26+c29+c30+c31) + 2*(c4-c3-c2+c5) + 3*(-c7-c8+c17+c18) + 6*(-c23-c24+c27+c28) + 10*(-c10-c11+c14+c15)));

    g.z =   u2000*((c3-c2-c4+c5-c7+c8-c17+c18-c20+c21-c30+c31-c34+c35-c36+c37) + 2*(-c9+c12-c13+c16-c22+c25-c26+c29) + 8*(-c10+c11-c14+c15-c23+c24-c27+c28)) +
          8*u0020*((c11-c9-c13+c15-c22+c24-c26+c28)) +
          4*(u0200*((c8-c7-c9+c12-c14+c15-c20+c21-c22+c25-c27+c28) + 3*(-c10+c11-c23+c24)) +
             u0002*((c3-c2-c4+c5-c9+c12-c13+c16-c23+c24-c27+c28) + 3*(-c10+c11-c14+c15)) +
             u1*(u2*((c3-c4-c7+c8-c13-c20+c21-c26+c35-c36) + 2*(-c14-c27) + 3*(-c9-c22) + 4*(-c10+c15-c23+c28) + 6*(c11+c24)) +
                 u3*((c3-c2-c4+c5-c7+c8-c13+c16-c20+c21-c22+c25) + 2*(-c9+c12-c27+c28) + 4*(-c14+c15-c23+c24) + 6*(-c10+c11))) +
             u3*(((c3-c2-c4+c5+c8+c18-c20-c22-c26-c30) + 2*(-c23-c27) + 3*(-c9-c13) + 4*(-c10-c14+c24+c28) + 6*(c11+c15))*u2)) +
          2*(u0*(u1*((c3-c2-c4+c5-c13+c16-c26+c29-c34+c35-c36+c37) + 2*(-c7+c8-c20+c21) + 3*(-c9+c12-c22+c25) + 6*(-c14+c15-c27+c28) + 10*(-c10+c11-c23+c24)) +
                 u2*((c3-c2-c4+c5-c7+c8-c17+c18-c20+c21-c30+c31-c34+c35-c36+c37) + 4*(-c9-c13-c22-c26) + 6*(-c10-c14-c23-c27) + 10*(c11+c15+c24+c28)) +
                 u3*((-c7+c8-c17+c18-c20+c21-c22+c25-c26+c29-c30+c31) + 2*(c3-c2-c4+c5) + 3*(-c9+c12-c13+c16) + 6*(-c23+c24-c27+c28) + 10*(-c10+c11-c14+c15))));

    return g;
}


// xx yy zz yz zx xy
vec3[2] eval_H_expr_red(void)
{
    vec3[2] h;

    h[0].x = 2*u0*((c0+c1+c3+c4-c6-c8-c13-c15+c19+c20+c21+c22+c24+c26+c27+c28) + 2*(-c7-c9-c11-c14) + 4*(c2-c10)) +
               u1*((c0+c4+c6-c7-c8+c13-c14-c15+c19-c20-c21+c26-c27-c28+c32+c36) + 2*(c3-c9-c22+c35) + 4*(c2-c10-c23+c34)) +
               u2*((c1+c3+c6+c8-c9-c11-c13-c15+c19+c21-c22-c24-c26-c28+c33+c35) + 2*(c4-c7-c20+c36) + 4*(c2-c10-c23+c34)) +
               u3*((c0+c1+c3+c4-c7-c9-c11-c14-c20-c22-c24-c27+c32+c33+c35+c36) + 4*(c2-c10-c23+c34));
    h[0].y = 2*u0*((c0-c1-c3+c4+c6+c8+c13+c15+c19+c20+c21-c22-c24+c26+c27+c28) + 2*(c7-c9-c11+c14) + 4*(-c10-c23)) +
               u1*((c0+c4+c6+c13+c19+c26+c32+c36) + 2*(-c3-c9-c22-c35) + 3*(c7+c8+c14+c15+c20+c21+c27+c28) + 4*(-c11-c24) + 8*(-c10-c23)) +
               u2*((c1+c3+c6+c8-c9-c11-c13-c15+c19+c21-c22-c24-c26-c28+c33+c35) + 2*(-c4+c7+c20-c36) + 4*(-c10+c17-c23+c30)) +
               u3*((c0-c1-c3+c4+c32-c33-c35+c36) + 2*(c6+c8+c13+c15+c19+c21+c26+c28) + 3*(c7-c9-c11+c14+c20-c22-c24+c27) + 8*(-c10-c23));
    h[0].z = 2*u0*((c1-c0+c3-c4+c6+c8+c13+c15+c19-c20+c21+c22+c24+c26-c27+c28) + 2*(-c7+c9+c11-c14) + 4*(-c10-c23)) +
               u1*((c0+c4+c6-c7-c8+c13-c14-c15+c19-c20-c21+c26-c27-c28+c32+c36) + 2*(-c3+c9+c22-c35) + 4*(-c10+c12-c23+c25)) +
               u2*((c1+c3+c6+c8+c19+c21+c33+c35) + 2*(-c4-c7-c20-c36) + 3*(c9+c11+c13+c15+c22+c24+c26+c28) + 4*(-c14-c27) + 8*(-c10-c23)) +
               u3*((c1-c0+c3-c4-c32+c33+c35-c36) + 2*(c6+c8+c13+c15+c19+c21+c26+c28) + 3*(-c7+c9+c11-c14-c20+c22+c24-c27) + 8*(-c10-c23));
    h[1].x = 2*u0*(((c6-c8-c13+c15+c19-c21-c26+c28)) +
                u3*((c6-c8-c13+c15+c19-c21-c26+c28))) +
               u1*((c0-c4+c6+c7-c13-c14+c19+c20-c26-c27+c32-c36) + 3*(-c8+c15-c21+c28)) +
               u2*((c1-c3+c6-c8+c9-c11+c19-c21+c22-c24+c33-c35) + 3*(-c13+c15-c26+c28));
    h[1].y = 2*u0*((c1-c3-c19+c21-c22+c24-c26+c28) + 2*(c9-c11)) +
               u1*((c0+c4+c6+c7-c8+c13+c14-c15-c19-c20+c21-c26-c27+c28-c32-c36) + 2*(-c3+c9-c22+c35) + 4*(-c11+c24)) +
               u2*((c1-c3+c6-c8+c13-c15-c19+c21-c26+c28-c33+c35) + 3*(c9-c11-c22+c24)) +
               u3*((c1-c3+c6-c8+c13-c15-c19+c21-c26+c28-c33+c35) + 3*(c9-c11-c22+c24));
    h[1].z = 2*u0*((c0-c4-c19-c20-c21+c26+c27+c28) + 2*(c7-c14)) +
               u1*((c0-c4+c6+c8-c13-c15-c19-c21+c26+c28-c32+c36) + 3*(c7-c14-c20+c27)) +
               u2*((c1+c3+c6+c8+c9+c11-c13-c15-c19-c21-c22-c24+c26+c28-c33-c35) + 2*(-c4+c7-c20+c36) + 4*(-c14+c27)) +
               u3*((c0-c4+c6+c8-c13-c15-c19-c21+c26+c28-c32+c36) + 3*(c7-c14-c20+c27));

    return h;
}

vec3[2] eval_H_expr_green(void)
{
    vec3[2] h;

    h[0].x = 2*u0*((c0+c1+c3+c4+c5-c6-c7-c8-c9-c13-c15+c20+c21+c22+c23+c24+c26+c27+c28) + 3*(c2-c11-c14) + 4*(-c10)) +
               u1*((c0+c4+c5+c6-c8-c9+c13-c15+c16-c20-c25+c26-c27+c36) + 2*(c3-c11-c14-c22-c23-c28) + 3*(c2+c35) + 4*(-c10+c34)) +
               u2*((c1+c3+c5+c6-c7+c8-c13-c15+c18+c21-c22-c24-c30+c35) + 2*(c4-c11-c14-c20-c23-c28) + 3*(c2+c36) + 4*(-c10+c34)) +
               u3*((c0+c1-c7-c9-c16-c18+c20+c21+c22+c25+c26+c30) + 2*(-c8-c13+c23+c28) + 3*(c3+c4+c24+c27) + 4*(c2+c5-c15) + 6*(-c11-c14) + 8*(-c10));
    h[0].y = 2*u0*((c0-c1-c2-c3+c4+c5+c6+c8-c9+c13+c14+c15+c20+c21-c22-c24+c26+c27+c28) + 3*(c7-c11-c23) + 4*(-c10)) +
               u1*((c0-c2+c4+c5+c6-c9+c13+c16-c25+c26-c35+c36) + 2*(-c3+c14-c22+c28) + 3*(c8+c15+c20+c27) + 4*(c7+c21-c24) + 6*(-c11-c23) + 8*(-c10)) +
               u2*((c1-c2+c3+c5+c6+c8-c13-c15+c18+c21-c22-c24+c35-c36) + 2*(-c4-c11-c14+c20-c23-c28) + 3*(c7+c30) + 4*(-c10+c17)) +
               u3*((c0+c1-c3-c4-c9-c16+c20+c21+c22-c24+c25+c26-c27+c30) + 2*(c8-c11-c13-c14-c23-c28) + 3*(c7+c18) + 4*(-c10+c17));
    h[0].z = 2*u0*((c1-c0-c2+c3-c4+c5+c6-c7+c8+c11+c13+c15-c20+c21+c22+c24+c26-c27+c28) + 3*(c9-c14-c23) + 4*(-c10)) +
               u1*((c0-c2+c4+c5+c6-c8+c13-c15+c16-c20+c26-c27-c35+c36) + 2*(-c3-c11-c14+c22-c23-c28) + 3*(c9+c25) + 4*(-c10+c12)) +
               u2*((c1-c2+c3+c5+c6-c7+c8+c18+c21-c30+c35-c36) + 2*(-c4+c11-c20+c28) + 3*(c13+c15+c22+c24) + 4*(c9+c26-c27) + 6*(-c14-c23) + 8*(-c10)) +
               u3*((c0+c1-c3-c4-c7-c18+c20+c21+c22-c24+c25+c26-c27+c30) + 2*(-c8-c11+c13-c14-c23-c28) + 3*(c9+c16) + 4*(-c10+c12));
    h[1].x = 2*u0*((c5-c2+c6+c7-c8+c9-c11-c13-c14+c15-c21+c23-c26+c28)) +
               u1*((c0-c2-c4+c5+c6+c9-c13+c16+c20-c25-c26-c27+c35-c36) + 2*(c7-c11-c14-c21+c23+c28) + 3*(-c8+c15)) +
               u2*((c1-c2-c3+c5+c6+c7-c8+c18-c21+c22-c24-c30-c35+c36) + 2*(c9-c11-c14+c23-c26+c28) + 3*(-c13+c15)) +
               u3*((c0+c1-c3-c4+c7+c9+c16+c18+c20-c21+c22-c24-c25-c26-c27-c30) + 2*(-c8-c11-c13-c14+c23+c28) + 4*c15);
    h[1].y = 2*u0*((c1+c2-c3-c5-c7+c9-c11+c14+c21-c22-c23+c24-c26+c28)) +
               u1*((c0+c2+c4-c5+c6-c8+c9+c13-c15-c16-c20+c25-c26-c27+c35-c36) + 2*(-c3-c11+c14-c22-c23+c28) + 4*c24) +
               u2*((c1+c2-c3-c5+c6-c7-c8+c13-c15-c18+c21+c30+c35-c36) + 2*(c9-c11+c14-c23-c26+c28) + 3*(-c22+c24)) +
               u3*((c0+c1+c4-c7+c9-c16+c18-c20+c21-c22+c25-c26-c27-c30) + 2*(c2-c5-c11+c14-c23+c28) + 3*(-c3+c24));
    h[1].z = 2*u0*((c0+c2-c4-c5+c7-c9+c11-c14-c20-c21-c23+c26+c27+c28)) +
               u1*((c0+c2-c4-c5+c6+c8-c9-c13-c15-c16+c25+c26-c35+c36) + 2*(c7+c11-c14-c21-c23+c28) + 3*(-c20+c27)) +
               u2*((c1+c2+c3-c5+c6+c7+c8-c13-c15-c18-c21-c22-c24+c30-c35+c36) + 2*(-c4+c11-c14-c20-c23+c28) + 4*c27) +
               u3*((c0+c1+c3+c7-c9+c16-c18-c20-c21-c22-c24-c25+c26+c30) + 2*(c2-c5+c11-c14-c23+c28) + 3*(-c4+c27));

    return h;
}

vec3[2] eval_H_expr_blue(void)
{
    vec3[2] h;

    h[0].x = 2*u0*((c2+c3+c4+c5-c10-c11-c14-c15-c23-c24-c27-c28+c34+c35+c36+c37)) +
               u1*((c4+c5-c9-c12+c13+c16-c22-c25+c26+c29+c36+c37) + 2*(-c10-c11-c14-c15-c23-c24-c27-c28) + 3*(c2+c3+c34+c35)) +
               u2*((c3+c5-c7+c8-c17+c18-c20+c21-c30+c31+c35+c37) + 2*(-c10-c11-c14-c15-c23-c24-c27-c28) + 3*(c2+c4+c34+c36)) +
               u3*((-c7-c8-c9-c12-c13-c16-c17-c18+c20+c21+c22+c25+c26+c29+c30+c31) + 2*(c23+c24+c27+c28) + 4*(c2+c3+c4+c5) + 6*(-c10-c11-c14-c15));
    h[0].y = 2*u0*((c7+c8-c10-c11-c14-c15+c17+c18+c20+c21-c23-c24-c27-c28+c30+c31)) +
               u1*((c4-c3-c2+c5-c9-c12+c13+c16-c22-c25+c26+c29-c34-c35+c36+c37) + 2*(c14+c15+c27+c28) + 4*(c7+c8+c20+c21) + 6*(-c10-c11-c23-c24)) +
               u2*((c3-c2-c4+c5+c8+c18+c21+c31-c34+c35-c36+c37) + 2*(-c10-c11-c14-c15-c23-c24-c27-c28) + 3*(c7+c17+c20+c30)) +
               u3*((-c9-c12-c13-c16+c20+c21+c22+c25+c26+c29+c30+c31) + 2*(-c10-c11-c14-c15-c23-c24-c27-c28) + 3*(c7+c8+c17+c18));
    h[0].z = 2*u0*((c9-c10-c11+c12+c13-c14-c15+c16+c22-c23-c24+c25+c26-c27-c28+c29)) +
               u1*((c4-c3-c2+c5+c13+c16+c26+c29-c34-c35+c36+c37) + 2*(-c10-c11-c14-c15-c23-c24-c27-c28) + 3*(c9+c12+c22+c25)) +
               u2*((c3-c2-c4+c5-c7+c8-c17+c18-c20+c21-c30+c31-c34+c35-c36+c37) + 2*(c11+c15+c24+c28) + 4*(c9+c13+c22+c26) + 6*(-c10-c14-c23-c27)) +
               u3*((-c8-c7-c17-c18+c20+c21+c22+c25+c26+c29+c30+c31) + 2*(-c10-c11-c14-c15-c23-c24-c27-c28) + 3*(c9+c12+c13+c16));
    h[1].x = u0*((c7-c8+c9-c12-c13+c16-c17+c18+c20-c21+c22-c25-c26+c29-c30+c31) + 2*(c10-c11-c14+c15+c23-c24-c27+c28)) +
             u1*((c3-c2-c4+c5+c9-c12-c13+c16+c22-c25-c26+c29-c34+c35-c36+c37) + 2*(c7-c8+c10-c11-c14+c15+c20-c21+c23-c24-c27+c28)) +
             u2*((c4-c3-c2+c5+c7-c8-c17+c18+c20-c21-c30+c31-c34-c35+c36+c37) + 2*(c9+c10-c11-c13-c14+c15+c22+c23-c24-c26-c27+c28)) +
             u3*((c7-c8+c9-c12-c13+c16-c17+c18+c20-c21+c22-c25-c26+c29-c30+c31) + 2*(c10-c11-c14+c15+c23-c24-c27+c28));
    h[1].y = u0*((c2-c3+c4-c5+c9-c12+c13-c16-c22+c25-c26+c29-c34+c35-c36+c37) + 2*(c10-c11+c14-c15-c23+c24-c27+c28)) +
             u1*((c2-c3+c4-c5+c9-c12+c13-c16-c22+c25-c26+c29-c34+c35-c36+c37) + 2*(c10-c11+c14-c15-c23+c24-c27+c28)) +
             u2*((c2-c3+c4-c5-c7-c8-c17-c18+c20+c21+c30+c31-c34+c35-c36+c37) + 2*(c9+c10-c11+c13+c14-c15-c22-c23+c24-c26-c27+c28)) +
             u3*((-c7+c8+c9-c12+c13-c16-c17+c18-c20+c21-c22+c25-c26+c29-c30+c31) + 2*(c2-c3+c4-c5+c10-c11+c14-c15-c23+c24-c27+c28));
    h[1].z = u0*((c2+c3-c4-c5+c7+c8-c17-c18-c20-c21+c30+c31-c34-c35+c36+c37) + 2*(c10+c11-c14-c15-c23-c24+c27+c28)) +
             u1*((c2+c3-c4-c5-c9-c12-c13-c16+c22+c25+c26+c29-c34-c35+c36+c37) + 2*(c7+c8+c10+c11-c14-c15-c20-c21-c23-c24+c27+c28)) +
             u2*((c2+c3-c4-c5+c7+c8-c17-c18-c20-c21+c30+c31-c34-c35+c36+c37) + 2*(c10+c11-c14-c15-c23-c24+c27+c28)) +
             u3*((c7+c8-c9-c12+c13+c16-c17-c18-c20-c21-c22-c25+c26+c29+c30+c31) + 2*(c2+c3-c4-c5+c10+c11-c14-c15-c23-c24+c27+c28));

    return h;
}

vec3 compute_gradient(vec3 dummy)
{
    vec3 g = int(type_tet==TYPE_RED) * eval_G_expr_red() +
             int(type_tet==TYPE_GREEN) * eval_G_expr_green() +
             int(type_tet==TYPE_BLUE) * eval_G_expr_blue();

    // permutation
    if(type_P == 1) g = g.zxy;
    else if(type_P == 2) g = g.yzx;

    // reflection
    g *= vec3(type_R);

    // scale
    return (DENOM_G*scale_norm)*g;
}


float[6] compute_Hessian(vec3 dummy)
{
    vec3[2] h;
    switch(type_tet){
        case TYPE_RED   : h = eval_H_expr_red();   break;
        case TYPE_GREEN : h = eval_H_expr_green(); break;
        case TYPE_BLUE  : h = eval_H_expr_blue();  break;
    }

    // permutation
    if(type_P == 1) {
        h[0] = h[0].zxy;
        h[1] = h[1].zxy;
    }
    else if(type_P == 2) {
        h[0] = h[0].yzx;
        h[1] = h[1].yzx;
    }

    // reflection
    h[1] *= float(type_R.x*type_R.y*type_R.z)*vec3(type_R);

    // scale
    h[0] *= scale_norm*scale_norm*DENOM_H;
    h[1] *= scale_norm.yzx*scale_norm.zxy*DENOM_H;

    return float[6](h[0].x, h[0].y, h[0].z, h[1].x, h[1].y, h[1].z);
}

#define	SHADING_BLINN_PHONG 1
struct TMaterial
{
	vec3	ambient;
	vec3	diffuse;
	vec3	specular;
	vec3	emission;
	float	shininess;
};
struct TLight
{
	vec4	position;
	vec3	ambient;
	vec3	diffuse;
	vec3	specular;
};

TLight		uLight = TLight(
        vec4(1,1,1,0),
        vec3(.2,.2,.2),
        vec3(1,1,1),
        vec3(1,1,1)
        );

vec4 shade_Blinn_Phong(vec3 n, vec4 pos_eye, TMaterial material, TLight light)
{
	vec3	l;
	if(light.position.w == 1.0)
		l = normalize((light.position - pos_eye).xyz);		// positional light
	else
		l = normalize((light.position).xyz);	// directional light
	vec3	v = -normalize(pos_eye.xyz);
	vec3	h = normalize(l + v);
	float	l_dot_n = max(dot(l, n), 0.0);
	vec3	ambient = light.ambient * material.ambient;
	vec3	diffuse = light.diffuse * material.diffuse * l_dot_n;
	vec3	specular = vec3(0.0);

	if(l_dot_n >= 0.0)
	{
		specular = light.specular * material.specular * pow(max(dot(h, n), 0.0), material.shininess);
	}
	return vec4(ambient + diffuse + specular, 1);
}


vec4 compute_color(vec4 p, vec3 g, float[6] d2) {

	float	Dxx = d2[0];
	float	Dyy = d2[1];
	float	Dzz = d2[2];
	float	Dyz = d2[3];
	float	Dzx = d2[4];
	float	Dxy = d2[5];

	mat3	H = mat3(Dxx, Dxy, Dzx,
					Dxy, Dyy, Dyz,
					Dzx, Dyz, Dzz);
	float	one_over_len_g = 1.0/length(g);
	vec3	n = -g*one_over_len_g;
	mat3    P = mat3(1.0) - mat3(n.x*n.x, n.x*n.y, n.x*n.z,
								n.x*n.y, n.y*n.y, n.y*n.z,
								n.x*n.z, n.y*n.z, n.z*n.z);
	mat3    M = -P*H*P*one_over_len_g;
	float   T = M[0][0] + M[1][1] + M[2][2];
	mat3    MMt = M*transpose(M);
	float   F = sqrt(MMt[0][0] + MMt[1][1] + MMt[2][2]);
	float   k_max = (T + sqrt(2.0*F*F - T*T))*0.5;
	float   k_min = (T - sqrt(2.0*F*F - T*T))*0.5;

	float	scale_k = 10;
	vec2	tc = vec2(scale_k*vec2(k_max,k_min)+0.5);

	if(p.w!=0.0)
	{
#if	SHADING_BLINN_PHONG
		TMaterial	material = 
			TMaterial(
				vec3(.1,.1,.1),
				texture(tex_colormap_2d, tc).xyz,
				vec3(1,1,1),
				vec3(0,0,0),
				128.0*0.5
				);
		return shade_Blinn_Phong(normalize(mat3(MV)*(-p.w*g)), MV*vec4(p.xyz,1), material, uLight);
#else
		return texture(tex_colormap_2d, tc);
#endif
	}
}


void main() {

    vec3 start = texture(tex_front, vTexCoord).xyz*scale_lattice + offset_lattice;
    vec3 end = texture(tex_back, vTexCoord).xyz*scale_lattice + offset_lattice;
    
    vec3    p = start;
    vec3    p_prev;
    vec3    dir = normalize(end-start);
    
    float   step = scale_step*dim_max;
    
    float   len = 0;
    float   len_full = length(end - start);
    float   voxel, voxel_prev;

    voxel = EVAL(p);

    float   orientation = 2.0*float(voxel < level)-1.0;	// equivalent to (voxel<level?1:-1)

    for(int i = 0 ; i < MAX_ITERATIONS ; i++)
    {
        p += step*dir;
        len += step;
        if(len > len_full)
        {
            break;
        }
        
        voxel = EVAL(p);
        
        if(orientation*voxel > orientation*level)
        {
            if(abs(voxel-voxel_prev) > 0.00001)
            {
                p = (p*(voxel_prev-level) - p_prev*(voxel-level))/(voxel_prev-voxel);
#ifndef NO_PREPROCESS
                preprocess(p);
#endif
#ifndef NO_FETCH_COEFFICIENTS
                fetch_coefficients();
#endif
            }

#ifdef PREPROC_DERIVATIVES
            preproc_derivatives(p);
#endif
            vec3    g = compute_gradient(p);
            float[6] H = compute_Hessian(p);

            vec3        pos = (p*vec3(scale_norm) + vec3(offset_norm) - vec3(.5,.5,.5));
            fColor = compute_color(vec4(pos,orientation), g, H);
            return;
        }
        voxel_prev = voxel;
        p_prev = p;
    }
    fColor = vec4(1,1,1,1);
}

