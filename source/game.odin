/*
This file is the starting point of your game.

Some importants procedures:
- game_init: Initializes sokol_gfx and sets up the game state.
- game_frame: Called one per frame, do your game logic and rendering in here.
- game_cleanup: Called on shutdown of game, cleanup memory etc.

The hot reload compiles the contents of this folder into a game DLL. A host
application loads that DLL and calls the procedures of the DLL. 

Special procedures that help facilitate the hot reload:
- game_memory: Run just before a hot reload. The hot reload host application can
	that way keep a pointer to the game's memory and feed it to the new game DLL
	after the hot reload is complete.
- game_hot_reloaded: Sets the `g` global variable in the new game DLL. The value
	comes from the value the host application got from game_memory before the
	hot reload.

When release or web builds are made, then this whole package is just
treated as a normal Odin package. No DLL is created.

The hot applications use sokol_app to open the window. They use the settings
returned by the `game_app_default_desc` procedure.
*/
package game

import "core:fmt"
import "core:os"
import "core:image/png"
import "core:strings"
import "core:math/linalg"
import "core:log"
import "core:slice"
import sapp "sokol/app"
import sg "sokol/gfx"
import sglue "sokol/glue"
import slog "sokol/log"

import "gltf2"

Game_Memory :: struct {
	pip: sg.Pipeline,
	bind: sg.Bindings,
	rx, ry: f32,
	index_count: i32,
	camera_pos: Vec3,
	camera_target: Vec3,
	camera_zoom: f32,
	is_rotating: bool,
	last_mouse_pos: [2]f32,
	data: gltf2.Data
}

Mat4 :: matrix[4,4]f32
Vec3 :: [3]f32
g: ^Game_Memory

Vertex :: struct {
	x, y, z: f32,
	color: u32,
	u, v: u16,
}

Node_Transform :: struct {
    global_transform: Mat4, // 4x4 transformation matrix
}



@export
game_app_default_desc :: proc() -> sapp.Desc {
	return {
		width = 1280,
		height = 720,
		sample_count = 4,
		window_title = "Odin + Sokol GLTF Renderer",
		icon = { sokol_default = true },
		logger = { func = slog.func },
		html5_update_document_title = true,
	}
}

@export
game_init :: proc() {    
	g = new(Game_Memory)
    game_hot_reloaded(g)

    sg.setup({
        environment = sglue.environment(),
        logger = { func = slog.func },
        buffer_pool_size = 2048,
    })

    // Load GLB file
    gltf_data, gltf_error := gltf2.load_from_file("assets/st101.glb")
    switch err in gltf_error {
    case gltf2.JSON_Error:
        log.error("JSON parsing error:", err)
    case gltf2.GLTF_Error:
        log.error("GLTF parsing error:", err)
    }
    defer gltf2.unload(gltf_data)

    // Dynamic arrays for all vertices and indices
    all_vertices: [dynamic]Vertex
    all_indices: [dynamic]u16

    // Traverse the node hierarchy and compute global transforms
    transforms: [dynamic]Node_Transform
    for node_index in gltf_data.scenes[gltf_data.scene.?].nodes {
        traverse_nodes(gltf_data.nodes, int(node_index), linalg.MATRIX4F32_IDENTITY, &transforms)
    }

    // Process meshes with their node transforms
    process_meshes_with_transforms(gltf_data, transforms[:], &all_vertices, &all_indices)

    // Create buffers and render
    if len(all_vertices) == 0 || len(all_indices) == 0 {
        log.error("No valid mesh data found. Vertex or index data is empty.")
        return
    }

    // Create a single vertex buffer and index buffer
    vertex_buffer := sg.make_buffer({
        type = .VERTEXBUFFER,
        usage = .DYNAMIC,
        size = uint(size_of(Vertex) * len(all_vertices)),
    })
    if vertex_buffer.id == 0 {
        log.error("Failed to create vertex buffer")
        return
    }

    index_buffer := sg.make_buffer({
        type = .INDEXBUFFER,
        usage = .DYNAMIC,
        size = uint(size_of(u16) * len(all_indices)),
    })
    if index_buffer.id == 0 {
        log.error("Failed to create index buffer")
        return
    }

    // Update the buffers with the actual data
    sg.update_buffer(vertex_buffer, {
        ptr = raw_data(all_vertices),
        size = uint(size_of(Vertex) * len(all_vertices)),
    })

    sg.update_buffer(index_buffer, {
        ptr = raw_data(all_indices),
        size = uint(size_of(u16) * len(all_indices)),
    })

    // Bind the buffers
    g.bind.vertex_buffers[0] = vertex_buffer
    g.bind.index_buffer = index_buffer
    g.index_count = i32(len(all_indices))
	// g.scene = gltf_data.scene

	// Load texture and create sampler
	if img_data, img_data_ok := read_entire_file("assets/maindiffuse.png", context.temp_allocator); img_data_ok {
		if img, img_err := png.load_from_bytes(img_data, allocator = context.temp_allocator); img_err == nil {
			// Create the image
			g.bind.images[IMG_tex] = sg.make_image({
				width = i32(img.width),
				height = i32(img.height),
				data = {
					subimage = {
						0 = {
							0 = { ptr = raw_data(img.pixels.buf), size = uint(slice.size(img.pixels.buf[:])) },
						},
					},
				},
			})

			g.bind.samplers[SMP_smp] = sg.make_sampler({
                wrap_u = .REPEAT,
                wrap_v = .REPEAT,
            })
		} else {
			log.error(img_err)
		}
	} else {
		log.error("Failed loading texture")
	}

	// Create pipeline with the shader
	shader := sg.make_shader(texcube_shader_desc(sg.query_backend()))
	if shader.id == 0 {
		log.error("Failed to create shader")
		return
	}

	g.pip = sg.make_pipeline({
		shader = shader,
		layout = {
			attrs = {
				ATTR_texcube_pos = { format = .FLOAT3 },
				ATTR_texcube_color0 = { format = .UBYTE4N },
				ATTR_texcube_texcoord0 = { format = .SHORT2N },
			},
		},
		index_type = .UINT16,
		cull_mode = .NONE,
		depth = {
			compare = .LESS_EQUAL,
			write_enabled = true,
		},
	})
	if g.pip.id == 0 {
		log.error("Failed to create pipeline")
		return
	}

	// Initialize camera state
	g.camera_pos = {1500.0, 300.0, 500.0}
	g.camera_target = {-210.648525, 38.5765915, 412.4304745}
	g.camera_zoom = 1.0
	g.is_rotating = false
	g.last_mouse_pos = {0.0, 0.0}
}

@export
game_frame :: proc() {
	dt := f32(sapp.frame_duration())
	g.rx += 60 * dt
	g.ry += 100 * dt

	// Compute MVP matrix
	vs_params := Vs_Params {
		mvp = compute_mvp(g.rx, g.ry, g.camera_pos, g.camera_target, g.camera_zoom),
	}

	// Clear screen and render
	pass_action := sg.Pass_Action {
		colors = {
			0 = { load_action = .CLEAR, clear_value = { 0.41, 0.68, 0.83, 1 } },
		},
		depth = { load_action = .CLEAR, clear_value = 1.0 },
	}

	// Begin pass
	sg.begin_pass({ action = pass_action, swapchain = sglue.swapchain() })

	// Apply pipeline
	sg.apply_pipeline(g.pip)

	// Apply bindings
	sg.apply_bindings(g.bind)

	// Apply uniforms
	sg.apply_uniforms(UB_vs_params, { ptr = &vs_params, size = size_of(vs_params) })

    sg.draw(0, g.index_count, 1)
	// Draw the model
	// for node in g.data.nodes {
    //     mesh := g.data.meshes[node.mesh.?]
    //     for i in 0..< len(mesh.primitives) {
    //         primitive := mesh.primitives[i]
    //         sg.draw(0, primitive.indices.?, 1)
    //     }
	// }

	// End pass
	sg.end_pass()

	// Commit the frame
	sg.commit()

	free_all(context.temp_allocator)
}

compute_mvp :: proc(rx, ry: f32, camera_pos, camera_target: Vec3, zoom: f32) -> matrix[4,4]f32 {
	proj := linalg.matrix4_perspective(60.0 * linalg.RAD_PER_DEG, sapp.widthf() / sapp.heightf(), 0.01, 10.0)
	view := linalg.matrix4_look_at_f32({0.0, -1.5, -6.0}, {}, {0.0, 1.0, 0.0})
	view_proj := proj * view
	rxm := linalg.matrix4_rotate_f32(rx * linalg.RAD_PER_DEG, {1.0, 0.0, 0.0})
	rym := linalg.matrix4_rotate_f32(ry * linalg.RAD_PER_DEG, {0.0, 1.0, 0.0})
	model := rxm * rym
	return view_proj * model
}

NIL_MATRIX :: gltf2.Matrix4{} // All zeros


is_nil_matrix :: proc(mat: gltf2.Matrix4) -> bool {
    return mat == NIL_MATRIX
}

traverse_nodes :: proc(nodes: []gltf2.Node, node_index: int, parent_transform: Mat4, transforms: ^[dynamic]Node_Transform) {
    if node_index < 0 || node_index >= len(nodes) {
        return // Invalid node index
    }

    node := nodes[node_index]

    // Compute the local transform matrix
    local_transform: Mat4
	if !is_nil_matrix(node.mat) {
        // Use the provided matrix if available
        local_transform = node.mat
    } else {
        // Compute the local transform from translation, rotation, and scale
        translation := linalg.matrix4_translate_f32(node.translation)
        rotation := linalg.matrix4_from_quaternion_f32(node.rotation)
        scale := linalg.matrix4_scale_f32(node.scale)
        local_transform = translation * rotation * scale
    }

    // Compute the global transform
    global_transform := parent_transform * local_transform

    // Store the global transform
    append(transforms, Node_Transform{global_transform = global_transform})

    // Recursively process children
    for child_index in node.children {
        traverse_nodes(nodes, cast(int)child_index, global_transform, transforms)
    }
}

Vec4 :: [4]f32

process_meshes_with_transforms :: proc(gltf_data: ^gltf2.Data, transforms: []Node_Transform, all_vertices: ^[dynamic]Vertex, all_indices: ^[dynamic]u16) {
    g.data = gltf_data^
    // Pre-allocate memory for vertices and indices
    estimated_vertex_count := 0
    estimated_index_count := 0
    for node in gltf_data.nodes {
        if node.mesh == nil do continue
        mesh := gltf_data.meshes[node.mesh.?]
        for primitive in mesh.primitives {
            if strings.contains(mesh.name.?, "main") && strings.contains(mesh.name.?, "Sub_0") && !strings.contains(mesh.name.?, "Other") {
                log.info(mesh.name)
                pos_attr_index, has_pos_attr := primitive.attributes["POSITION"]
                assert(has_pos_attr, "Mesh has no position attribute")
                pos_attr := gltf_data.accessors[pos_attr_index]
                xyz_buf := gltf2.buffer_slice(gltf_data, pos_attr.buffer_view.?).([][3]f32)            
                estimated_vertex_count += len(xyz_buf)*3
                index_attr_index, has_index_attr := primitive.indices.?
                assert(has_index_attr, "Mesh has no index attribute")
                index_attr := gltf_data.accessors[index_attr_index]
                index_buf := gltf2.buffer_slice(gltf_data, index_attr.buffer_view.?).([]u16)
            }
        }
    }
    reserve(all_vertices, len(all_vertices) + estimated_vertex_count)
    reserve(all_indices, len(all_indices) + estimated_index_count)

    // Process nodes
    for node, node_index in gltf_data.nodes {
        if node.mesh == nil do continue // Skip nodes without a mesh

        // Get the global transform for this node
        global_transform := transforms[node_index].global_transform

        // Get the mesh data
        mesh := gltf_data.meshes[node.mesh.?]

        // Process the mesh's primitives
        for primitive in mesh.primitives {
            if strings.contains(mesh.name.?, "main") && strings.contains(mesh.name.?, "Sub_0") && !strings.contains(mesh.name.?, "Other") {
                // Extract vertex data (POSITION, NORMAL, etc.)
                pos_attr_index, has_pos_attr := primitive.attributes["POSITION"]
                assert(has_pos_attr, "Mesh has no position attribute")
                pos_attr := gltf_data.accessors[pos_attr_index]
                xyz_buf := gltf2.buffer_slice(gltf_data, pos_attr.buffer_view.?).([][3]f32)

                // Transform the vertices using the global transform
                for i in 0..<len(xyz_buf) {
                    pos := xyz_buf[i]
                    transformed_pos := linalg.matrix_mul_vector(global_transform, Vec4{pos[0], pos[1], pos[2], 1.0})
                    xyz_buf[i] = {transformed_pos.x, transformed_pos.y, transformed_pos.z}
                }

                // Extract UV data (if available)
                uv_buf: [][2]f32
                if uv_attr_index, has_uv_attr := primitive.attributes["TEXCOORD_0"]; has_uv_attr {
                    uv_attr := gltf_data.accessors[uv_attr_index]
                    uv_buf = gltf2.buffer_slice(gltf_data, uv_attr.buffer_view.?).([][2]f32)
                } else {
                    // Default UVs if not present
                    uv_buf = make([][2]f32, len(xyz_buf))
                    defer delete(uv_buf)
                    for i in 0..<len(uv_buf) {
                        uv_buf[i] = {0.0, 0.0}
                    }
                }

                // Extract normal data (if available)
                normal_buf: [][3]f32
                if normal_attr_index, has_normal_attr := primitive.attributes["NORMAL"]; has_normal_attr {
                    normal_attr := gltf_data.accessors[normal_attr_index]
                    normal_buf = gltf2.buffer_slice(gltf_data, normal_attr.buffer_view.?).([][3]f32)
                } else {
                    // Default normals if not present
                    normal_buf = make([][3]f32, len(xyz_buf))
                    defer delete(normal_buf)
                    for i in 0..<len(normal_buf) {
                        normal_buf[i] = {0.0, 1.0, 0.0} // Default normal pointing up
                    }
                }

                // Extract color data (if available)
                color_buf: [][4]f32
                if color_attr_index, has_color_attr := primitive.attributes["COLOR_0"]; has_color_attr {
                    color_attr := gltf_data.accessors[color_attr_index]
                    #partial switch color_attr.component_type {
                    case .Unsigned_Byte: // Color data is in [4]u8 format
                        u8_color_buf := gltf2.buffer_slice(gltf_data, color_attr.buffer_view.?).([][4]u8)
                        color_buf = make([][4]f32, len(u8_color_buf))
                        defer delete(color_buf)
                        for i in 0..<len(u8_color_buf) {
                            color := u8_color_buf[i]
                            color_buf[i] = {
                                f32(color[0]) / 255.0, // Normalize to [0, 1]
                                f32(color[1]) / 255.0,
                                f32(color[2]) / 255.0,
                                f32(color[3]) / 255.0,
                            }
                        }
                    case .Float: // Color data is already in [4]f32 format
                        color_buf = gltf2.buffer_slice(gltf_data, color_attr.buffer_view.?).([][4]f32)
                    case: // Unsupported color format
                        log.error("Unsupported color format:", color_attr.component_type)
                        color_buf = make([][4]f32, len(xyz_buf))
                        defer delete(color_buf)
                        for i in 0..<len(color_buf) {
                            color_buf[i] = {1.0, 1.0, 1.0, 1.0} // Default white color
                        }
                    }
                } else {
                    // Default color if not present
                    color_buf = make([][4]f32, len(xyz_buf))
                    defer delete(color_buf)
                    for i in 0..<len(color_buf) {
                        color_buf[i] = {1.0, 1.0, 1.0, 1.0} // Default white color
                    }
                }

                // Extract index data
                index_attr_index, has_index_attr := primitive.indices.?
                assert(has_index_attr, "Mesh has no index attribute")
                index_attr := gltf_data.accessors[index_attr_index]
                index_buf := gltf2.buffer_slice(gltf_data, index_attr.buffer_view.?).([]u16)

                // Append vertices and indices to the dynamic arrays
                for i in 0..<len(xyz_buf) {
                    pos := xyz_buf[i]
                    uv := uv_buf[i]
                    normal := normal_buf[i]
                    color := color_buf[i]

                    append(all_vertices, Vertex{
                        x = pos[0],
                        y = pos[1],
                        z = pos[2],
                        color = u32(color[0] * 255) << 24 | u32(color[1] * 255) << 16 | u32(color[2] * 255) << 8 | u32(color[3] * 255), // Pack color into u32
                        u = u16(uv[0] * 32767),
                        v = u16(uv[1] * 32767),
                    })
                }

                for idx in index_buf {
                    append(all_indices, idx)
                }
            }
        }
    }
}

@export
game_cleanup :: proc() {
	sg.shutdown()
	free(g)
}

force_reset: bool

@export
game_event :: proc(e: ^sapp.Event) {
	#partial switch e.type {
	case .KEY_DOWN:
		if e.key_code == .F6 {
			force_reset = true
		}
	case .MOUSE_DOWN:
		if e.mouse_button == .LEFT {
			g.is_rotating = true
			g.last_mouse_pos = {e.mouse_x, e.mouse_y}
		}
	case .MOUSE_UP:
		if e.mouse_button == .LEFT {
			g.is_rotating = false
		}
	case .MOUSE_SCROLL:
		g.camera_zoom += e.scroll_y * 0.1
		g.camera_zoom = max(0.1, g.camera_zoom) // Ensure zoom doesn't go below 0.1
	case .MOUSE_MOVE:
		if g.is_rotating {
			dx := e.mouse_x - g.last_mouse_pos[0]
			dy := e.mouse_y - g.last_mouse_pos[1]
			g.last_mouse_pos = {e.mouse_x, e.mouse_y}

			// Adjust camera position based on mouse movement
			g.camera_pos.x += dx * 2.0
			g.camera_pos.y -= dy * 2.0
		}
	}
}


@(export)
game_memory :: proc() -> rawptr {
	return g
}

@(export)
game_memory_size :: proc() -> int {
	return size_of(Game_Memory)
}

@(export)
game_hot_reloaded :: proc(mem: rawptr) {
	g = (^Game_Memory)(mem)
}

@(export)
game_force_restart :: proc() -> bool {
	return false
}