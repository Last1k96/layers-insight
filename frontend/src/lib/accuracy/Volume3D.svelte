<script lang="ts">
	import { getSpatialDims, formatValue } from './tensorUtils';

	let { main, ref, shape, mainLabel = 'Main', refLabel = 'Reference' }: {
		main: Float32Array;
		ref: Float32Array;
		shape: number[];
		mainLabel?: string;
		refLabel?: string;
	} = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let container: HTMLDivElement | undefined = $state();
	let containerW = $state(800);
	let containerH = $state(600);
	let rotX = $state(-25);
	let rotY = $state(30);
	let threshold = $state(0.1);
	let opacity = $state(0.3);

	let dragging = $state(false);
	let dragStartX = $state(0);
	let dragStartY = $state(0);
	let rotStartX = $state(0);
	let rotStartY = $state(0);

	let userZoom = $state(1.0);

	// Tooltip state
	let tooltip = $state<{ x: number; y: number; ci: number; iy: number; ix: number; mainVal: number; refVal: number; diffVal: number } | null>(null);

	// Render params shared between render effect and mousemove
	let renderParams = $state<{
		scale: number; cosX: number; sinX: number; cosY: number; sinY: number;
		centerX: number; centerY: number; depthSpreadX: number; depthSpreadY: number;
		C: number; H: number; W: number; depthExtent: number; viewZ: number;
	} | null>(null);

	// ResizeObserver for container sizing
	$effect(() => {
		if (!container) return;
		const obs = new ResizeObserver((entries) => {
			for (const entry of entries) {
				containerW = entry.contentRect.width;
				containerH = entry.contentRect.height;
			}
		});
		obs.observe(container);
		return () => obs.disconnect();
	});

	// Magma-style colormap stops
	const MAGMA_STOPS: [number, number, number][] = [
		[0, 0, 0],
		[80, 18, 123],
		[182, 55, 100],
		[230, 107, 57],
		[252, 194, 68],
		[252, 253, 191],
	];

	function sampleColormap(t: number): [number, number, number] {
		const clamped = Math.max(0, Math.min(1, t));
		const segment = clamped * (MAGMA_STOPS.length - 1);
		const idx = Math.min(Math.floor(segment), MAGMA_STOPS.length - 2);
		const local = segment - idx;
		const a = MAGMA_STOPS[idx];
		const b = MAGMA_STOPS[idx + 1];
		return [
			Math.round(a[0] + (b[0] - a[0]) * local),
			Math.round(a[1] + (b[1] - a[1]) * local),
			Math.round(a[2] + (b[2] - a[2]) * local),
		];
	}

	let volumeData = $derived.by(() => {
		const dims = getSpatialDims(shape);
		const { channels: C, height: H, width: W } = dims;

		const diff = new Float32Array(C * H * W);
		let minVal = Infinity;
		let maxVal = -Infinity;

		for (let i = 0; i < C * H * W; i++) {
			const v = Math.abs(main[i] - ref[i]);
			diff[i] = v;
			if (v < minVal) minVal = v;
			if (v > maxVal) maxVal = v;
		}

		if (maxVal === minVal) maxVal = minVal + 1;

		return { diff, C, H, W, minVal, maxVal };
	});

	function onMouseDown(e: MouseEvent) {
		dragging = true;
		dragStartX = e.clientX;
		dragStartY = e.clientY;
		rotStartX = rotX;
		rotStartY = rotY;
		tooltip = null;
	}

	function onMouseMove(e: MouseEvent) {
		if (dragging) {
			rotY = rotStartY + (e.clientX - dragStartX) * 0.5;
			rotX = rotStartX + (e.clientY - dragStartY) * 0.5;
			rotX = Math.max(-89, Math.min(89, rotX));
			tooltip = null;
			return;
		}

		// Hit-test for tooltip
		if (!canvas || !renderParams) return;
		const rect = canvas.getBoundingClientRect();
		const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
		const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);

		const rp = renderParams;
		const { C, H, W, scale, cosY, sinY, cosX, sinX, centerX, centerY, depthSpreadX, depthSpreadY, viewZ } = rp;
		const { diff, minVal, maxVal } = volumeData;
		const span = maxVal - minVal || 1;
		const invSpan = 1 / span;
		const frontToBack = viewZ >= 0;

		// Iterate slices front-to-back (reverse of render order) to find topmost hit
		for (let si = C - 1; si >= 0; si--) {
			const ci = frontToBack ? (C - 1 - si) : si;
			const z = C > 1 ? ci / (C - 1) : 0.5;
			const zCentered = z - 0.5;

			const sx = centerX + zCentered * depthSpreadX - (W * scale * cosY) / 2;
			const sy = centerY + zCentered * depthSpreadY - (H * scale) / 2;

			// Inverse transform: screen -> slice pixel coords
			const localX = (mouseX - sx) / (scale * cosY);
			const localY = (mouseY - sy) / scale;

			const ix = Math.floor(localX);
			const iy = Math.floor(localY);

			if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;

			const v = diff[ci * H * W + iy * W + ix];
			const norm = (v - minVal) * invSpan;

			if (norm < threshold) continue;

			// Hit found
			const mainIdx = ci * H * W + iy * W + ix;
			tooltip = {
				x: e.clientX,
				y: e.clientY,
				ci, iy, ix,
				mainVal: main[mainIdx],
				refVal: ref[mainIdx],
				diffVal: v,
			};
			return;
		}
		tooltip = null;
	}

	function onMouseUp() {
		dragging = false;
	}

	function onMouseLeave() {
		dragging = false;
		tooltip = null;
	}

	function onWheel(e: WheelEvent) {
		e.preventDefault();
		userZoom *= e.deltaY > 0 ? 0.9 : 1.1;
		userZoom = Math.max(0.1, Math.min(10, userZoom));
	}

	// 3D projection helper: rotate point and project to 2D
	function project3D(
		x: number, y: number, z: number,
		cosX: number, sinX: number, cosY: number, sinY: number,
		centerX: number, centerY: number, scale: number,
	): [number, number] {
		// Rotate around Y axis
		const rx = x * cosY - z * sinY;
		const rz = x * sinY + z * cosY;
		// Rotate around X axis
		const ry = y * cosX - rz * sinX;
		return [centerX + rx * scale, centerY + ry * scale];
	}

	$effect(() => {
		if (!canvas) return;
		const { diff, C, H, W, minVal, maxVal } = volumeData;
		const _rotX = rotX;
		const _rotY = rotY;
		const _threshold = threshold;
		const _opacity = opacity;
		const _cw = containerW;
		const _ch = containerH;

		const cw = Math.max(1, Math.floor(_cw));
		const ch = Math.max(1, Math.floor(_ch));
		canvas.width = cw;
		canvas.height = ch;

		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		// Clear with dark background
		ctx.fillStyle = '#111827';
		ctx.fillRect(0, 0, cw, ch);

		if (C === 0 || H === 0 || W === 0) return;

		const span = maxVal - minVal || 1;
		const invSpan = 1 / span;

		// Projection parameters
		const cosX = Math.cos(_rotX * Math.PI / 180);
		const sinX = Math.sin(_rotX * Math.PI / 180);
		const cosY = Math.cos(_rotY * Math.PI / 180);
		const sinY = Math.sin(_rotY * Math.PI / 180);

		const depthExtent = C > 1 ? C : 0;

		// Rotation-independent base scale (worst-case diagonal)
		const diagonal = Math.sqrt(W * W + H * H + depthExtent * depthExtent) || 1;
		const margin = 80;
		const _userZoom = userZoom;
		const baseScale = Math.min((cw - margin) / diagonal, (ch - margin) / diagonal);
		const scale = baseScale * _userZoom;

		// Volume is centered at origin, so centroid is always (0,0)
		const centerX = cw / 2;
		const centerY = ch / 2;

		// Depth spread matching project3D convention: z contributes -sinY to X, -cosY*sinX to Y
		const depthSpreadX = -depthExtent * sinY * scale;
		const depthSpreadY = -depthExtent * cosY * sinX * scale;

		const viewZ = cosX * cosY;
		const frontToBack = viewZ >= 0;

		// Store render params for mousemove hit-testing
		renderParams = { scale, cosX, sinX, cosY, sinY, centerX, centerY, depthSpreadX, depthSpreadY, C, H, W, depthExtent, viewZ };

		const offscreen = new OffscreenCanvas(W, H);
		const offCtx = offscreen.getContext('2d');
		if (!offCtx) return;

		// Render slices back-to-front
		for (let si = 0; si < C; si++) {
			const ci = frontToBack ? (C - 1 - si) : si;
			const z = C > 1 ? ci / (C - 1) : 0.5;

			const imgData = offCtx.createImageData(W, H);
			const pixels = imgData.data;
			const sliceOffset = ci * H * W;

			let hasVisiblePixel = false;

			for (let y = 0; y < H; y++) {
				for (let x = 0; x < W; x++) {
					const v = diff[sliceOffset + y * W + x];
					const norm = (v - minVal) * invSpan;

					if (norm < _threshold) continue;

					hasVisiblePixel = true;
					const [r, g, b] = sampleColormap(norm);
					const pxIdx = (y * W + x) * 4;
					pixels[pxIdx] = r;
					pixels[pxIdx + 1] = g;
					pixels[pxIdx + 2] = b;
					pixels[pxIdx + 3] = Math.round(255 * _opacity * norm);
				}
			}

			if (!hasVisiblePixel) continue;

			offCtx.putImageData(imgData, 0, 0);

			const zCentered = z - 0.5;
			const sx = centerX + zCentered * depthSpreadX - (W * scale * cosY) / 2;
			const sy = centerY + zCentered * depthSpreadY - (H * scale) / 2;

			ctx.save();
			ctx.globalAlpha = 1;
			ctx.setTransform(
				scale * cosY, 0,
				0, scale,
				sx, sy
			);
			ctx.drawImage(offscreen, 0, 0);
			ctx.restore();
		}

		// --- Draw axes along volume edges ---
		// Volume-to-screen using slice convention (matches setTransform rendering)
		function volToScreen(vx: number, vy: number, vz: number): [number, number] {
			return [
				centerX + ((vx - W / 2) * cosY - vz * sinY) * scale,
				centerY + ((vy - H / 2) - vz * cosY * sinX) * scale,
			];
		}

		// Fixed axes at the (0, 0, 0) corner — origin of the volume
		const origin: [number, number, number] = [0, 0, -depthExtent / 2];

		const edgeAxes: { from: [number, number, number]; to: [number, number, number]; label: string; color: string; dimSize: number }[] = [
			{ from: origin, to: [W, 0, -depthExtent / 2], label: 'W', color: '#ef4444', dimSize: W },
			{ from: origin, to: [0, H, -depthExtent / 2], label: 'H', color: '#22c55e', dimSize: H },
			{ from: origin, to: [0, 0, depthExtent / 2], label: 'C', color: '#3b82f6', dimSize: C },
		];

		for (const axis of edgeAxes) {
			if (axis.label === 'C' && C <= 1) continue;

			const [p1x, p1y] = volToScreen(...axis.from);
			const [p2x, p2y] = volToScreen(...axis.to);

			// Draw axis line
			ctx.strokeStyle = axis.color;
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			ctx.moveTo(p1x, p1y);
			ctx.lineTo(p2x, p2y);
			ctx.stroke();

			// Tick marks (cap at 10)
			const numTicks = Math.min(10, axis.dimSize);
			const tickLen = 5;
			const dx = p2x - p1x;
			const dy = p2y - p1y;
			const len = Math.sqrt(dx * dx + dy * dy) || 1;
			const perpX = -dy / len;
			const perpY = dx / len;

			for (let ti = 0; ti <= numTicks; ti++) {
				const t = ti / numTicks;
				const tx2 = p1x + dx * t;
				const ty2 = p1y + dy * t;
				ctx.strokeStyle = axis.color;
				ctx.lineWidth = 1;
				ctx.beginPath();
				ctx.moveTo(tx2, ty2);
				ctx.lineTo(tx2 + perpX * tickLen, ty2 + perpY * tickLen);
				ctx.stroke();
			}

			// Label dim size at the far end
			ctx.fillStyle = axis.color;
			const labelX = p2x + (p2x - p1x) * 0.06 + perpX * 12;
			const labelY = p2y + (p2y - p1y) * 0.06 + perpY * 12;
			ctx.font = '11px monospace';
			ctx.fillText(`${axis.label}: ${axis.dimSize}`, labelX, labelY);
		}

		// --- Draw corner gizmo (secondary reference) ---
		const axisLen = Math.min(cw, ch) * 0.1;
		const axisOriginX = 40;
		const axisOriginY = ch - 40;

		const gizmoAxes: { dir: [number, number, number]; label: string; color: string }[] = [
			{ dir: [1, 0, 0], label: 'W', color: '#ef4444' },
			{ dir: [0, 1, 0], label: 'H', color: '#22c55e' },
			{ dir: [0, 0, 1], label: 'C', color: '#3b82f6' },
		];

		for (const axis of gizmoAxes) {
			const [adx, ady, adz] = axis.dir;
			const [endX, endY] = project3D(
				adx, -ady, adz,
				cosX, sinX, cosY, sinY,
				axisOriginX, axisOriginY, axisLen,
			);

			ctx.strokeStyle = axis.color;
			ctx.lineWidth = 2;
			ctx.beginPath();
			ctx.moveTo(axisOriginX, axisOriginY);
			ctx.lineTo(endX, endY);
			ctx.stroke();

			ctx.fillStyle = axis.color;
			ctx.beginPath();
			ctx.arc(endX, endY, 3, 0, Math.PI * 2);
			ctx.fill();

			const labelOffX = (endX - axisOriginX) * 0.2;
			const labelOffY = (endY - axisOriginY) * 0.2;
			ctx.font = '11px monospace';
			ctx.textAlign = 'center';
			ctx.textBaseline = 'middle';
			ctx.fillText(axis.label, endX + labelOffX, endY + labelOffY);
		}

		ctx.fillStyle = '#9ca3af';
		ctx.beginPath();
		ctx.arc(axisOriginX, axisOriginY, 2, 0, Math.PI * 2);
		ctx.fill();

		// Draw info text and colorbar
		ctx.fillStyle = '#9ca3af';
		ctx.font = '12px monospace';
		ctx.textAlign = 'left';
		ctx.textBaseline = 'top';
		ctx.fillText(`Volume: ${C} x ${H} x ${W}`, 8, 8);
		ctx.fillText(`Diff range: ${formatValue(minVal)} - ${formatValue(maxVal)}`, 8, 24);

		// Mini colorbar
		const barX = cw - 170;
		const barY = 10;
		const barW = 120;
		const barH = 12;
		for (let px = 0; px < barW; px++) {
			const t = px / (barW - 1);
			const [r, g, b] = sampleColormap(t);
			ctx.fillStyle = `rgb(${r},${g},${b})`;
			ctx.fillRect(barX + px, barY, 1, barH);
		}
		ctx.strokeStyle = '#666';
		ctx.lineWidth = 1;
		ctx.strokeRect(barX, barY, barW, barH);
		ctx.fillStyle = '#9ca3af';
		ctx.font = '10px monospace';
		ctx.textAlign = 'left';
		ctx.fillText(formatValue(minVal), barX, barY + barH + 3);
		ctx.textAlign = 'right';
		ctx.fillText(formatValue(maxVal), barX + barW, barY + barH + 3);
	});
</script>

<div class="h-full flex flex-col gap-2">
	<div class="flex flex-wrap items-center gap-4 text-xs text-gray-400 shrink-0">
		<label class="flex items-center gap-1.5">
			<span class="whitespace-nowrap">Hide errors below</span>
			<input
				type="range"
				min="0"
				max="1"
				step="0.01"
				bind:value={threshold}
				class="w-24 accent-purple-500"
			/>
			<span class="w-8 text-right font-mono text-gray-300">{threshold.toFixed(2)}</span>
		</label>
		<label class="flex items-center gap-1.5">
			<span class="whitespace-nowrap">Slice opacity</span>
			<input
				type="range"
				min="0.05"
				max="1"
				step="0.05"
				bind:value={opacity}
				class="w-24 accent-purple-500"
			/>
			<span class="w-8 text-right font-mono text-gray-300">{opacity.toFixed(2)}</span>
		</label>
		<span class="text-gray-500 italic">Drag to orbit · Scroll to zoom</span>
	</div>
	<div bind:this={container} class="flex-1 min-h-0 relative">
		<canvas
			bind:this={canvas}
			class="w-full h-full rounded border border-gray-700 cursor-grab"
			class:cursor-grabbing={dragging}
			onmousedown={onMouseDown}
			onmousemove={onMouseMove}
			onmouseup={onMouseUp}
			onmouseleave={onMouseLeave}
			onwheel={onWheel}
		></canvas>
		{#if tooltip}
			<div
				class="fixed z-50 pointer-events-none px-3 py-2 rounded-lg border border-gray-600 bg-gray-800/95 text-xs font-mono text-gray-200 shadow-xl"
				style="left: {tooltip.x + 14}px; top: {tooltip.y + 14}px;"
			>
				<div class="text-purple-300 font-semibold mb-1">[C={tooltip.ci}, H={tooltip.iy}, W={tooltip.ix}]</div>
				<div><span class="text-gray-400">{mainLabel}:</span> {formatValue(tooltip.mainVal)}</div>
				<div><span class="text-gray-400">{refLabel}:</span> {formatValue(tooltip.refVal)}</div>
				<div><span class="text-yellow-400">Diff:</span> <span class="text-yellow-300">{formatValue(tooltip.diffVal)}</span></div>
			</div>
		{/if}
	</div>
</div>
