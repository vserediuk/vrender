#version 450

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference : require

#include "input_structures.glsl"

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;

struct Vertex {

	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 color;
	ivec4 inJointIndices;
	vec4 inJointWeights;
}; 

layout(buffer_reference, std430) readonly buffer VertexBuffer{ 
	Vertex vertices[];
};

layout(std430, set = 1, binding = 0) readonly buffer JointMatrices {
	mat4 jointMatrices[];
};

//push constants block
layout( push_constant ) uniform constants
{
	mat4 render_matrix;
	VertexBuffer vertexBuffer;
} PushConstants;

void main() 
{
	Vertex v = PushConstants.vertexBuffer.vertices[gl_VertexIndex];
	
	vec4 position = vec4(v.position, 1.0f);

	gl_Position =  sceneData.viewproj * PushConstants.render_matrix *position;

	vec4 weights = v.inJointWeights / dot(v.inJointWeights, vec4(1.0));
        mat4 skinMat = 
        weights.x * jointMatrices[int(v.inJointIndices.x)] +
        weights.y * jointMatrices[int(v.inJointIndices.y)] +
        weights.z * jointMatrices[int(v.inJointIndices.z)] +
        weights.w * jointMatrices[int(v.inJointIndices.w)];

	outNormal = (PushConstants.render_matrix * skinMat * vec4(v.normal, 0.f)).xyz;
	outColor = v.color.xyz * materialData.colorFactors.xyz;	
	outUV.x = v.uv_x;
	outUV.y = v.uv_y;
}